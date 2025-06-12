import json
import os
import sys
import argparse
import torch
import librosa
import numpy as np
from pydub import AudioSegment
from audiocraft.models import MusicGen
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import logging
import traceback
import scipy.io.wavfile
from pathlib import Path
import glob
import tempfile
import random

# Wyłącz ostrzeżenia xFormers i inne
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*xFormers.*")
os.environ['XFORMERS_MORE_DETAILS'] = '0'

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('music_gen.log'),
        logging.StreamHandler()
    ]
)

def extract_audio_features(audio_path, sr=22050, duration=30):
    """Ekstraktuj cechy muzyczne z pliku audio używając librosa"""
    try:
        # Wczytanie audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Podstawowe cechy
        features = {}
        
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo.item())  # Extract scalar using .item()
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc'] = np.mean(mfcc, axis=1).tolist()
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = float(np.mean(zero_crossing_rate))
        
        # Chroma features (dla tonacji)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma'] = np.mean(chroma, axis=1).tolist()
        
        # Próba określenia tonacji
        chroma_mean = np.mean(chroma, axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_mean)
        features['key'] = key_names[key_idx] + ' major'
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy'] = float(np.mean(rms))
        
        # Oszacowanie dancability na podstawie regularności rytmu
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        if len(onset_times) > 1:
            onset_intervals = np.diff(onset_times)
            rhythm_regularity = 1.0 - np.std(onset_intervals) / np.mean(onset_intervals)
            features['danceability'] = float(max(0, min(1, rhythm_regularity)))
        else:
            features['danceability'] = 0.5
        
        # Energy (na podstawie RMS)
        features['energy'] = float(min(1.0, features['rms_energy'] * 10))
        
        # Valence (na podstawie brightnessu - spectral centroid)
        # Wyższe częstotliwości = bardziej pozytywny
        normalized_centroid = min(1.0, features['spectral_centroid'] / 4000.0)
        features['valence'] = float(normalized_centroid)
        
        logging.info(f"Ekstraktowano cechy dla {audio_path}")
        return features
        
    except Exception as e:
        logging.error(f"Błąd ekstraktowania cech dla {audio_path}: {e}")
        return None

def analyze_audio_directory(directory_path, extensions=None):
    """Analizuj wszystkie pliki audio w katalogu"""
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
    
    directory_path = Path(directory_path)
    if not directory_path.exists():
        logging.error(f"Katalog nie istnieje: {directory_path}")
        return {}
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(directory_path.glob(f'*{ext}'))
        audio_files.extend(directory_path.glob(f'*{ext.upper()}'))
    
    if not audio_files:
        logging.error(f"Nie znaleziono plików audio w katalogu: {directory_path}")
        return {}
    
    analyzed_files = {}
    print(f"\n🎵 Analizowanie {len(audio_files)} plików audio...")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"📊 Analizowanie {i}/{len(audio_files)}: {audio_file.name}")
        features = extract_audio_features(str(audio_file))
        if features:
            analyzed_files[str(audio_file)] = features
            print(f"   ✅ Tempo: {features['tempo']:.1f} BPM, Tonacja: {features['key']}")
        else:
            print(f"   ❌ Błąd analizy")
    
    return analyzed_files

def find_similar_track(analyzed_files, target_features=None):
    """Znajdź najbardziej podobny utwór lub wybierz losowy"""
    if not analyzed_files:
        return None, None
    
    if target_features is None:
        # Wybierz losowy utwór
        filepath = random.choice(list(analyzed_files.keys()))
        return filepath, analyzed_files[filepath]
    
    # Znajdź najbardziej podobny na podstawie cech
    min_distance = float('inf')
    best_match = None
    
    target_mfcc = np.array(target_features['mfcc'])
    
    for filepath, features in analyzed_files.items():
        current_mfcc = np.array(features['mfcc'])
        distance = np.linalg.norm(target_mfcc - current_mfcc)
        
        # Dodaj wagi dla innych cech
        tempo_diff = abs(features['tempo'] - target_features['tempo']) / 200.0
        energy_diff = abs(features['energy'] - target_features['energy'])
        
        total_distance = distance + tempo_diff + energy_diff
        
        if total_distance < min_distance:
            min_distance = total_distance
            best_match = (filepath, features)
    
    return best_match if best_match else (None, None)

def check_dependencies():
    """Sprawdź czy wszystkie potrzebne zależności są dostępne"""
    try:
        cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA dostępne: {cuda_available}")
        return True
    except Exception as e:
        logging.error(f"Błąd podczas sprawdzania zależności: {e}")
        return False

def normalize_features(analyzed_files):
    """Normalizuj cechy MFCC"""
    try:
        if not analyzed_files:
            return analyzed_files
            
        mfccs = [features['mfcc'] for features in analyzed_files.values()]
        scaler = StandardScaler()
        normalized_mfccs = scaler.fit_transform(mfccs)
        
        for i, filepath in enumerate(analyzed_files):
            analyzed_files[filepath]['normalized_mfcc'] = normalized_mfccs[i].tolist()
        
        logging.info("Znormalizowano cechy MFCC")
        return analyzed_files
    except Exception as e:
        logging.error(f"Błąd normalizacji: {e}")
        return analyzed_files

def load_and_preprocess_audio(filepath, duration=10, sr=32000):
    """Załaduj i przetwórz plik audio"""
    try:
        # Sprawdź rozszerzenie pliku
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(filepath)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(filepath)
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(filepath, format='flac')
        elif file_ext == '.m4a':
            audio = AudioSegment.from_file(filepath, format='m4a')
        elif file_ext == '.ogg':
            audio = AudioSegment.from_ogg(filepath)
        else:
            audio = AudioSegment.from_file(filepath)
        
        audio = audio[:duration * 1000]
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sr)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0
        return samples, sr
    except Exception as e:
        logging.error(f"Błąd przetwarzania audio {filepath}: {e}")
        return None, sr

def load_musicgen_model():
    """Załaduj model MusicGen z obsługą błędów"""
    try:
        logging.info("Ładowanie modelu MusicGen...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MusicGen.get_pretrained("facebook/musicgen-small")
        if hasattr(model, 'compression_model'):
            logging.info(f"Model MusicGen załadowany pomyślnie na {device}")
        else:
            logging.warning("Model MusicGen może nie być w pełni załadowany")
        return model, device
    except Exception as e:
        logging.error(f"Błąd ładowania modelu: {e}")
        logging.error(traceback.format_exc())
        raise

def create_structured_prompt(features, style_preference=None):
    """Utwórz strukturalny prompt z opisem przebiegu utworu"""
    try:
        key = features.get('key', 'C major')
        tempo = int(features.get('tempo', 120))
        energy = features.get('energy', 0.5)
        danceability = features.get('danceability', 0.5)
        valence = features.get('valence', 0.5)
        
        # Określ główny styl
        if style_preference:
            main_style = style_preference
        elif danceability > 0.7:
            main_style = "electronic dance"
        elif energy > 0.7:
            main_style = "energetic electronic"
        elif valence < 0.3:
            main_style = "ambient melancholic"
        elif valence > 0.7:
            main_style = "uplifting electronic"
        else:
            main_style = "electronic"
        
        # Określ strukturę utworu
        if tempo >= 120:
            structure = "intro with soft build-up, main section with driving beat, breakdown, climax with full energy, outro with fade"
        elif tempo >= 90:
            structure = "gentle intro, gradual build-up, main melody development, dynamic variation, smooth conclusion"
        else:
            structure = "ambient intro, slow melodic development, harmonic progression, textural evolution, peaceful ending"
        
        # Stwórz pełny prompt z kontekstem strukturalnym
        structured_prompt = f"{main_style} music in {key} at {tempo} BPM with complete song structure: {structure}"
        
        return structured_prompt
    except Exception as e:
        logging.error(f"Błąd tworzenia strukturalnego promptu: {e}")
        return f"Electronic music in {features.get('key', 'C major')} at {int(features.get('tempo', 120))} BPM"

def generate_full_track(model, device, prompt, features, duration=30, auto_prompt=False):
    """Generuj pełny utwór jako jedną całość z naturalnym przebiegiem"""
    try:
        # Przygotuj enhanced prompt
        if auto_prompt or prompt is None:
            enhanced_prompt = create_structured_prompt(features)
        else:
            key = features.get('key', 'C major')
            tempo = int(features.get('tempo', 120))
            
            # Wzbogać prompt o cechy muzyczne
            if 'bpm' not in prompt.lower() and 'tempo' not in prompt.lower():
                enhanced_prompt = f"{prompt}, {tempo} BPM"
            else:
                enhanced_prompt = prompt
                
            if key.lower() not in prompt.lower() and 'key' not in prompt.lower():
                enhanced_prompt = f"{enhanced_prompt} in {key}"
            
            # Dodaj strukturalny kontekst
            enhanced_prompt = f"{enhanced_prompt}, complete song with intro, development, and outro"
        
        logging.info(f"Generowanie pełnego utworu z promptem: {enhanced_prompt}")
        logging.info(f"Parametry: duration={duration}s, tempo={features.get('tempo')}, key={features.get('key')}")
        
        # Ustaw parametry generowania dla pełnego utworu
        model.set_generation_params(
            duration=duration,
            use_sampling=True,
            top_k=250,
            top_p=0.9,
            temperature=1.0
        )
        
        print(f"🎵 Generowanie pełnego utworu ({duration}s)...")
        print("⏳ To może potrwać kilka minut...")
        
        # Generuj pełny utwór jako jedną całość
        with torch.no_grad():
            audio = model.generate([enhanced_prompt], progress=True)
        
        if audio is not None and len(audio) > 0:
            generated_audio = audio[0].cpu().numpy()
            sample_rate = model.sample_rate
            
            # Sprawdź długość wygenerowanego audio
            actual_duration = generated_audio.shape[-1] / sample_rate
            logging.info(f"Wygenerowano pełny utwór: shape={generated_audio.shape}, duration={actual_duration:.1f}s, sr={sample_rate}")
            
            # Dodaj fade-in i fade-out dla naturalnego brzmienia
            generated_audio = apply_audio_effects(generated_audio, sample_rate)
            
            return generated_audio, sample_rate
        else:
            logging.error("Nie udało się wygenerować utworu")
            return None, None
            
    except Exception as e:
        logging.error(f"Błąd generowania pełnego utworu: {e}")
        logging.error(traceback.format_exc())
        return None, None

def apply_audio_effects(audio, sample_rate):
    """Zastosuj efekty audio dla lepszego brzmienia"""
    try:
        # Fade-in (pierwsze 2 sekundy)
        fade_in_samples = int(2.0 * sample_rate)
        if len(audio) > fade_in_samples:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in_curve
        
        # Fade-out (ostatnie 3 sekundy)
        fade_out_samples = int(3.0 * sample_rate)
        if len(audio) > fade_out_samples:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out_curve
        
        # Normalizacja głośności
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (0.95 / max_val)  # Normalizuj do 95% maksymalnej głośności
        
        logging.info("Zastosowano efekty audio: fade-in, fade-out, normalizacja")
        return audio
        
    except Exception as e:
        logging.error(f"Błąd aplikowania efektów audio: {e}")
        return audio

def extract_mfcc(audio, sr, n_mfcc=13):
    """Wyekstraktuj cechy MFCC z audio"""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        logging.error(f"Błąd ekstraktowania MFCC: {e}")
        return None

def create_auto_prompt(features, style_preference=None):
    """Utwórz automatyczny prompt na podstawie cech muzycznych z pełną strukturą"""
    try:
        key = features.get('key', 'C major')
        tempo = int(features.get('tempo', 120))
        danceability = features.get('danceability', 0.5)
        energy = features.get('energy', 0.5)
        valence = features.get('valence', 0.5)
        
        # Określ styl na podstawie cech lub preferencji
        if style_preference:
            style = style_preference
        elif danceability > 0.7:
            style = "dance"
        elif energy > 0.7:
            style = "energetic"
        elif valence < 0.3:
            style = "melancholic"
        elif valence > 0.7:
            style = "happy"
        else:
            style = "balanced"
        
        # Określ tempo
        if tempo < 90:
            tempo_desc = "slow"
        elif tempo < 120:
            tempo_desc = "medium"
        else:
            tempo_desc = "fast"
        
        energy_level = 'high' if energy > 0.7 else 'medium' if energy > 0.4 else 'low'
        
        # Utwórz prompt z pełną strukturą utworu
        auto_prompt = f"{style.capitalize()} electronic music in {key}, {tempo_desc} tempo ({tempo} BPM), {energy_level} energy, complete track with intro, main section, and outro"
        
        print(f"🤖 Automatyczny prompt: '{auto_prompt}'")
        return auto_prompt
    except Exception as e:
        logging.error(f"Błąd tworzenia automatycznego promptu: {e}")
        return "Electronic music with moderate tempo and balanced energy, complete track structure"

def get_user_prompt():
    """Pobierz prompt od użytkownika interaktywnie lub z argumentów"""
    print("\n" + "="*60)
    print("🎵 GENERATOR MUZYKI PODOBNEJ DO TWOJEJ 🎵")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='Generator muzyki podobnej do istniejącej')
    parser.add_argument('--dir', '-d', type=str, help='Katalog z plikami audio do analizy')
    parser.add_argument('--json', '-j', type=str, help='Plik JSON z wcześniej zapisanymi cechami')
    parser.add_argument('--prompt', '-p', type=str, help='Prompt opisujący pożądaną muzykę')
    parser.add_argument('--duration', type=int, default=30, help='Długość generowanej muzyki w sekundach (domyślnie 30)')
    parser.add_argument('--style', type=str, help='Preferowany styl muzyczny (electronic, ambient, energetic, dance, etc.)')
    parser.add_argument('--max-duration', type=int, default=120, help='Maksymalna długość utworu w sekundach (domyślnie 120)')
    parser.add_argument('--output', '-o', type=str, default='generated_music.wav', help='Nazwa pliku wyjściowego')
    parser.add_argument('--auto', '-a', action='store_true', help='Użyj automatycznego promptu na podstawie cech muzycznych')
    parser.add_argument('--similar', '-s', action='store_true', help='Generuj podobną muzykę na podstawie losowego utworu z katalogu')
    
    args = parser.parse_args()
    
    return args

def compare_features(generated_audio, sr, original_features):
    """Porównaj cechy wygenerowanego i oryginalnego audio"""
    try:
        generated_mfcc = extract_mfcc(generated_audio, sr)
        original_mfcc = np.array(original_features['mfcc'])
        if generated_mfcc is not None:
            distance = np.linalg.norm(generated_mfcc - original_mfcc)
            logging.info(f"Odległość MFCC między oryginałem a wygenerowanym: {distance:.4f}")
            return distance
        else:
            logging.error("Nie udało się wyekstraktować MFCC z wygenerowanego audio")
            return None
    except Exception as e:
        logging.error(f"Błąd porównywania cech: {e}")
        return None

def get_interactive_prompt():
    """Interaktywny interfejs do wyboru promptu"""
    print("\nWybierz sposób generowania muzyki:")
    print("1. 🎯 Wprowadź własny prompt (szczegółowy opis)")
    print("2. 🤖 Automatyczny prompt na podstawie cech muzycznych")
    print("3. 📋 Wybierz z gotowych szablonów")
    print("4. ❌ Wyjdź")
    
    while True:
        try:
            choice = input("\nTwój wybór (1-4): ").strip()
            if choice == '1':
                return get_custom_prompt(), False
            elif choice == '2':
                print("🤖 Będę używać automatycznego promptu na podstawie analizy twojej muzyki")
                return None, True
            elif choice == '3':
                return get_template_prompt(), False
            elif choice == '4':
                print("👋 Do widzenia!")
                sys.exit(0)
            else:
                print("❌ Nieprawidłowy wybór. Wybierz 1, 2, 3 lub 4.")
        except KeyboardInterrupt:
            print("\n👋 Do widzenia!")
            sys.exit(0)

def get_custom_prompt():
    """Pobierz niestandardowy prompt od użytkownika"""
    print("\n" + "-"*50)
    print("🎯 NIESTANDARDOWY PROMPT")
    print("-"*50)
    print("Opisz jakiej muzyki chcesz. Możesz użyć:")
    print("• Gatunek: electronic, rock, jazz, classical, pop")
    print("• Tempo: slow, medium, fast, lub BPM (np. 120 BPM)")
    print("• Nastrój: happy, sad, energetic, calm, aggressive")
    print("• Instrumenty: piano, guitar, drums, synthesizer")
    print("• Styl: ambient, upbeat, dreamy, powerful")
    
    while True:
        prompt = input("\n📝 Twój prompt: ").strip()
        if prompt:
            print(f"✅ Używam promptu: '{prompt}'")
            return prompt
        else:
            print("❌ Prompt nie może być pusty. Spróbuj ponownie.")

def get_template_prompt():
    """Wybierz prompt z gotowych szablonów"""
    templates = {
        '1': "Upbeat electronic dance music with heavy bass and energetic rhythm",
        '2': "Calm ambient music with soft synthesizers and dreamy atmosphere",
        '3': "Aggressive rock music with electric guitars and powerful drums",
        '4': "Smooth jazz with piano and saxophone, relaxing tempo",
        '5': "Classical orchestral music with strings and piano",
        '6': "Funky house music with groovy bass and rhythmic beats",
        '7': "Chill lo-fi hip hop with soft beats and warm atmosphere",
        '8': "Energetic pop music with catchy melody and upbeat tempo",
        '9': "Dark techno with industrial sounds and driving rhythm",
        '10': "Acoustic folk music with guitar and organic instruments"
    }
    
    print("\nDostępne szablony:")
    for key, value in templates.items():
        print(f"{key:2}. {value}")
    
    while True:
        try:
            choice = input("\nWybierz szablon (1-10): ").strip()
            if choice in templates:
                selected_prompt = templates[choice]
                print(f"✅ Wybrałeś: '{selected_prompt}'")
                return selected_prompt
            else:
                print("❌ Nieprawidłowy wybór. Wybierz numer od 1 do 10.")
        except KeyboardInterrupt:
            print("\n👋 Do widzenia!")
            sys.exit(0)

def main():
    """Główna funkcja programu"""
    try:
        args = get_user_prompt()
        
        if not check_dependencies():
            return
        
        # Określ źródło danych
        analyzed_files = {}
        
        if args.json:
            # Kompatybilność wsteczna - użyj istniejącego JSON
            print(f"📁 Wczytywanie danych z pliku JSON: {args.json}")
            with open(args.json, 'r', encoding='utf-8') as f:
                analyzed_files = json.load(f)
        elif args.dir:
            # Analizuj katalog
            print(f"📁 Analizowanie katalogu: {args.dir}")
            analyzed_files = analyze_audio_directory(args.dir)
            if not analyzed_files:
                print("❌ Nie znaleziono żadnych plików audio do analizy")
                return
        else:
            # Interaktywny tryb
            if not args.prompt and not args.auto and not args.similar:
                print("❌ Musisz podać --dir z katalogiem audio lub --json z plikiem cech")
                return
        
        if analyzed_files:
            analyzed_files = normalize_features(analyzed_files)
        
        # Określ prompt i parametry
        user_prompt = args.prompt
        auto_prompt = args.auto
        duration = min(args.duration, args.max_duration)  # Ogranicz maksymalną długość
        output_file = args.output
        style_preference = args.style
        
        if args.similar:
            # Tryb podobny - wybierz losowy utwór
            if not analyzed_files:
                print("❌ Brak danych do analizy podobieństwa")
                return
            
            filepath, features = find_similar_track(analyzed_files)
            if filepath and features:
                print(f"🎯 Wybrany utwór bazowy: {Path(filepath).name}")
                print(f"   📊 Tempo: {features['tempo']:.1f} BPM")
                print(f"   🎵 Tonacja: {features['key']}")
                print(f"   ⚡ Energia: {features['energy']:.2f}")
                auto_prompt = True
                user_prompt = create_structured_prompt(features, style_preference) if not auto_prompt else None
            else:
                print("❌ Nie udało się wybrać utworu bazowego")
                return
        elif not user_prompt and not auto_prompt:
            # Interaktywny tryb wyboru promptu
            user_prompt, auto_prompt = get_interactive_prompt()
            
            if not analyzed_files:
                print("❌ Brak danych muzycznych do analizy")
                return
            
            # Wybierz pierwszy dostępny utwór
            filepath = list(analyzed_files.keys())[0]
            features = analyzed_files[filepath]
        else:
            # Użyj pierwszego dostępnego utworu
            if analyzed_files:
                filepath = list(analyzed_files.keys())[0]
                features = analyzed_files[filepath]
            else:
                print("❌ Brak danych muzycznych")
                return
        
        logging.info("=== Rozpoczęcie generowania muzyki ===")
        logging.info(f"Parametry: prompt='{user_prompt}', duration={duration}s, output='{output_file}', auto={auto_prompt}, style={style_preference}")
        
        # Sprawdź czy długość nie jest zbyt duża
        if duration > 120:
            print(f"⚠️  Uwaga: Długość {duration}s może wymagać dużo czasu i pamięci")
            print("💡 Dla pierwszych testów zalecane jest 30-60 sekund")
        
        # Załaduj model
        model, device = load_musicgen_model()
        
        # Generuj muzykę
        logging.info(f"Używane cechy: tempo={features.get('tempo')}, key={features.get('key')}")
        
        generated_audio, sample_rate = generate_full_track(
            model, device, user_prompt, features, duration=duration, auto_prompt=auto_prompt
        )
        
        if generated_audio is not None:
            # Przygotuj audio do zapisu
            if len(generated_audio.shape) > 1:
                if generated_audio.shape[0] == 1:
                    audio_to_save = generated_audio.squeeze(0)
                elif generated_audio.shape[1] == 1:
                    audio_to_save = generated_audio.squeeze(1)
                else:
                    audio_to_save = generated_audio[0]
            else:
                audio_to_save = generated_audio
            
            # Zapisz plik
            try:
                sf.write(output_file, audio_to_save, sample_rate)
                logging.info(f"Zapisano wygenerowaną muzykę: {output_file}")
                
                # Weryfikacja
                test_audio, test_sr = sf.read(output_file)
                logging.info(f"Weryfikacja: wczytano {test_audio.shape} przy {test_sr} Hz")
                
                print(f"\n✅ Sukces! Wygenerowana muzyka została zapisana jako: {output_file}")
                print(f"🎵 Długość: {len(test_audio)/test_sr:.1f} sekund")
                print(f"🔊 Częstotliwość próbkowania: {test_sr} Hz")
                
                # Porównaj cechy
                distance = compare_features(audio_to_save, sample_rate, features)
                if distance is not None:
                    similarity = max(0, 100 - distance * 10)
                    print(f"📊 Podobieństwo do utworu bazowego: {similarity:.1f}%")
                
            except Exception as save_error:
                logging.error(f"Błąd zapisywania: {save_error}")
                print(f"❌ Nie udało się zapisać pliku: {save_error}")
                
        else:
            print("❌ Nie udało się wygenerować muzyki")
            
        logging.info("=== Generowanie zakończone ===")
        
    except Exception as e:
        logging.error(f"Błąd krytyczny: {e}")
        logging.error(traceback.format_exc())
        print(f"❌ Wystąpił błąd krytyczny: {e}")

if __name__ == "__main__":
    main()