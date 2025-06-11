import os
import json
import essentia
essentia.log.infoActive = False
import essentia.standard as es
import numpy as np
from scipy.spatial.distance import euclidean
from pathlib import Path
import psutil
import time
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QTextEdit, QFileDialog, QProgressBar,
    QSpinBox, QListWidget, QSlider, QDialog, QDoubleSpinBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsLineItem
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QMutex
from PyQt6.QtGui import QPen, QBrush, QColor
import sounddevice as sd
from pydub import AudioSegment
import logging
import gc
import traceback
from datetime import datetime
import math
from multiprocessing import Manager
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    filename='music_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Extract audio features
def extract_features(audio_file):
    try:
        sample_rate = 44100
        max_samples = int(30 * sample_rate)
        loader = es.MonoLoader(filename=audio_file, sampleRate=sample_rate)
        audio = loader()
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        frame_size = 2048
        hop_size = 1024
        spectrum = es.Spectrum()
        window = es.Windowing(type='hann')
        mfcc = es.MFCC(numberCoefficients=13)
        tempo = es.RhythmExtractor2013()
        key = es.KeyExtractor()
        danceability = es.Danceability()
        dynamic_complexity = es.DynamicComplexity()

        features = {
            'mfcc': [],
            'tempo': None,
            'key': None,
            'spectral_centroid': [],
            'energy': [],
            'danceability': None,
            'dynamic_complexity': None,
            'similarity_results': [],
            'duration': None
        }

        for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
            spec = spectrum(window(frame))
            mfcc_bands, mfcc_coeffs = mfcc(spec)
            features['mfcc'].append(mfcc_coeffs)
            features['spectral_centroid'].append(es.Centroid()(spec))
            features['energy'].append(es.Energy()(spec))

        features['mfcc'] = np.mean(features['mfcc'], axis=0).tolist()
        features['spectral_centroid'] = float(np.mean(features['spectral_centroid']))
        features['energy'] = float(np.mean(features['energy']))

        features['tempo'], _, _, _ = tempo(audio)
        key, scale, strength = key(audio)
        features['key'] = f"{key} {scale}"
        features['danceability'] = danceability(audio)[0]
        features['dynamic_complexity'] = dynamic_complexity(audio)[0]
        features['last_modified'] = os.path.getmtime(audio_file)
        features['duration'] = min(len(audio) / sample_rate, 30)

        del audio
        gc.collect()
        return features
    except Exception as e:
        logging.error(f"Error analyzing {audio_file}: {e}\n{traceback.format_exc()}")
        return None

# Extract features with retries
def extract_features_with_retry(audio_file, max_retries=3):
    for attempt in range(max_retries):
        try:
            return extract_features(audio_file)
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed for {audio_file}: {e}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to extract features for {audio_file} after {max_retries} attempts: {e}\n{traceback.format_exc()}")
                return None
            time.sleep(0.5)

# Calculate similarity
def calculate_similarity(features1, features2):
    try:
        mfcc_dist = euclidean(features1['mfcc'], features2['mfcc'])
        centroid_dist = abs(features1['spectral_centroid'] - features2['spectral_centroid'])
        energy_dist = abs(features1['energy'] - features2['energy'])
        tempo_dist = abs(features1['tempo'] - features2['tempo'])
        danceability_dist = abs(features1['danceability'] - features2['danceability'])
        dynamic_dist = abs(features1['dynamic_complexity'] - features2['dynamic_complexity'])
        total_distance = (0.4 * mfcc_dist + 0.2 * centroid_dist + 0.2 * energy_dist +
                          0.1 * tempo_dist + 0.05 * danceability_dist + 0.05 * dynamic_dist)
        return total_distance
    except Exception as e:
        logging.error(f"Error calculating similarity: {e}\n{traceback.format_exc()}")
        return float('inf')

# Get audio duration
def get_audio_duration(audio_file):
    try:
        loader = es.MonoLoader(filename=audio_file)
        audio = loader()
        sample_rate = loader.paramValue('sampleRate') or 44100
        duration = len(audio) / sample_rate if len(audio) > 0 else 180.0
        del audio
        gc.collect()
        logging.info(f"Calculated duration for {audio_file}: {duration}s")
        return duration
    except Exception as e:
        logging.error(f"Error calculating duration for {audio_file}: {e}\n{traceback.format_exc()}")
        return 180.0

# Calculate track energy
def calculate_track_energy(features):
    try:
        energy_weight = features['energy'] * 10
        danceability_weight = features['danceability'] * 0.5
        tempo_weight = (features['tempo'] - 60) / 200
        dynamic_weight = features['dynamic_complexity'] * 0.1
        spectral_weight = features['spectral_centroid'] * 0.05
        return energy_weight + danceability_weight + tempo_weight + dynamic_weight + spectral_weight
    except Exception as e:
        logging.error(f"Error calculating track energy: {e}\n{traceback.format_exc()}")
        return 0

# Calculate MFCC similarity
def calculate_mfcc_similarity(mfcc1, mfcc2):
    try:
        if not mfcc1 or not mfcc2 or len(mfcc1) != len(mfcc2):
            return 0
        mfcc1 = np.array(mfcc1, dtype=float)
        mfcc2 = np.array(mfcc2, dtype=float)
        dot_product = np.dot(mfcc1, mfcc2)
        norm1 = np.linalg.norm(mfcc1)
        norm2 = np.linalg.norm(mfcc2)
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0
    except Exception as e:
        logging.error(f"Error calculating MFCC: {e}\n{traceback.format_exc()}")
        return 0

# Check harmonic key
def is_harmonic_key(key1, key2):
    key_map = {
        'C major': ['G major', 'A minor', 'F major'],
        'A minor': ['C major', 'F major', 'D minor'],
        'C minor': ['G# major', 'D# minor', 'F minor'],
        'G# major': ['C minor', 'D# minor', 'A# major']
    }
    return key2 in key_map.get(key1, []) or key1 == key2

# Calculate compatibility
def calculate_compatibility(features1, features2, harmonic_mixing):
    try:
        tempo_diff = abs(features1['tempo'] - features2['tempo'])
        tempo_score = max(0, 1 - tempo_diff / 40)
        key_score = 1.0 if harmonic_mixing and is_harmonic_key(features1.get('key'), features2.get('key')) else 0.7
        energy1 = calculate_track_energy(features1)
        energy2 = calculate_track_energy(features2)
        energy_score = max(0, 1 - abs(energy1 - energy2) / 5)
        mfcc_score = calculate_mfcc_similarity(features1['mfcc'], features2['mfcc'])
        return (tempo_score * 0.3 + key_score * 0.25 + energy_score * 0.25 + mfcc_score * 0.2)
    except Exception as e:
        logging.error(f"Error calculating compatibility: {e}\n{traceback.format_exc()}")
        return 0.5

# Generate transition
def generate_transition(from_features, to_features, mix_config):
    try:
        compatibility = calculate_compatibility(from_features, to_features, mix_config['harmonic_mixing'])
        tempo_diff = abs(from_features['tempo'] - to_features['tempo'])
        transition_type = mix_config['transition_type']
        duration = mix_config['transition_duration']

        if transition_type == 'auto':
            if tempo_diff > 20:
                transition_type = 'beatmatch-slow'
                duration = max(12, duration)
            elif tempo_diff > 10:
                transition_type = 'beatmatch-medium'
                duration = max(8, duration)
            elif compatibility > 0.8:
                transition_type = 'quick-cut'
                duration = min(4, duration)
            else:
                transition_type = 'crossfade'

        technique = get_transition_technique(from_features, to_features, transition_type)
        return {
            'type': transition_type,
            'duration': duration,
            'compatibility': f"{compatibility:.2f}",
            'technique': technique,
            'fade_curve': mix_config['fade_curve']
        }
    except Exception as e:
        logging.error(f"Error generating transition: {e}\n{traceback.format_exc()}")
        return {
            'type': 'crossfade',
            'duration': 8,
            'compatibility': '0.50',
            'technique': 'Classic crossfade',
            'fade_curve': 'linear'
        }

# Get transition technique
def get_transition_technique(from_features, to_features, transition_type):
    from_energy = calculate_track_energy(from_features)
    to_energy = calculate_track_energy(to_features)
    if transition_type == 'beatmatch-slow':
        return 'Gradual acceleration with filter' if to_energy > from_energy else 'Gentle slowdown with echo'
    elif transition_type == 'beatmatch-medium':
        return 'Beat synchronization with pitch correction'
    elif transition_type == 'quick-cut':
        return 'Quick transition to breakdown'
    elif transition_type == 'echo':
        return 'Transition with echo effect'
    elif transition_type == 'high-pass':
        return 'Transition with high-pass filter'
    return 'Classic crossfade with EQ'

# Generate pool size and batch size
def get_dynamic_pool_size_and_batch(current_batch_size, total_files):
    max_cpu_percent = 95
    max_memory = 11
    memory_per_file = 0.2

    max_processes = int(psutil.cpu_count(logical=True) * 3)
    batch_size = max_processes

    batch_size = min(batch_size, total_files)

    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_used = psutil.virtual_memory().used / (1024 ** 3)
    available_memory = psutil.virtual_memory().available / (1024 ** 3)

    logging.info(f"Pool size: {max_processes}, batch size: {batch_size} "
                 f"(CPU: {cpu_percent:.1f}%, RAM: {memory_used:.1f} GB, Available RAM: {available_memory:.1f} GB)")
    return max_processes, batch_size

# Audio player thread
class AudioPlayerThread(QThread):
    update_time = pyqtSignal(float)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, tracks, start_index, volume, mix_config):
        super().__init__()
        self.tracks = tracks
        self.current_index = start_index
        self.volume = volume
        self.mix_config = mix_config
        self.is_playing = False
        self.current_time = 0
        self.mutex = QMutex()
        self.sample_rate = 44100
        self.stream = None

    def apply_transition(self, current_audio, next_audio, transition, duration, current_pos):
        t = current_pos / duration
        if t >= 1:
            return next_audio
        gain = t
        if transition['fade_curve'] == 'exponential':
            gain = t ** 2
        elif transition['fade_curve'] == 'logarithmic':
            gain = math.log1p(t * 9) / math.log(10)
        current_gain = (1 - gain) * self.volume
        next_gain = gain * self.volume
        return current_audio * current_gain + next_audio * next_gain

    def run(self):
        self.is_playing = True
        self.current_time = self.tracks[self.current_index]['start_time'] if self.tracks else 0

        def callback(out_data, frames, time_info, status):
            self.mutex.lock()
            try:
                if not self.is_playing or self.current_index >= len(self.tracks):
                    out_data.fill(0)
                    self.finished.emit()
                    return

                track = self.tracks[self.current_index]
                track_path = track['filepath']
                audio = AudioSegment.from_file(track_path).set_frame_rate(self.sample_rate).set_channels(2)
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                audio_data = audio_data.reshape(-1, 2)

                track_duration = track['play_duration']
                current_pos = self.current_time - track['start_time']
                sample_pos = int(current_pos * self.sample_rate)
                samples_needed = frames

                if sample_pos + samples_needed > len(audio_data):
                    samples_needed = len(audio_data) - sample_pos

                if samples_needed <= 0:
                    out_data.fill(0)
                    self.current_index += 1
                    self.current_time = track['start_time'] + track_duration
                    self.finished.emit()
                    return

                out_data[:samples_needed] = audio_data[sample_pos:sample_pos + samples_needed]
                out_data[samples_needed:] = 0

                if 'transition' in track and self.current_index + 1 < len(self.tracks):
                    transition_start = track_duration - track['transition']['duration']
                    if current_pos >= transition_start:
                        next_track = self.tracks[self.current_index + 1]
                        next_path = next_track['filepath']
                        next_audio = AudioSegment.from_file(next_path).set_frame_rate(self.sample_rate).set_channels(2)
                        next_data = np.array(next_audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                        next_data = next_data.reshape(-1, 2)
                        transition_samples = int(track['transition']['duration'] * self.sample_rate)
                        transition_pos = int((current_pos - transition_start) * self.sample_rate)
                        if transition_pos < len(next_data):
                            samples_to_take = min(samples_needed, len(next_data) - transition_pos)
                            out_data[:samples_to_take] = self.apply_transition(
                                out_data[:samples_to_take],
                                next_data[transition_pos:transition_pos + samples_to_take],
                                track['transition'],
                                track['transition']['duration'],
                                current_pos - transition_start
                            )

                self.current_time += frames / self.sample_rate
                self.update_time.emit(self.current_time)

                if current_pos >= track_duration:
                    self.current_index += 1
                    self.finished.emit()
            except Exception as e:
                self.error.emit(f"Audio playback error: {e}")
                out_data.fill(0)
            finally:
                self.mutex.unlock()

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=callback,
                blocksize=1024
            )
            self.stream.start()
            while self.is_playing and self.current_index < len(self.tracks):
                time.sleep(0.1)
        except Exception as e:
            self.error.emit(f"Stream error: {e}")
        finally:
            self.cleanup()

    def stop(self):
        self.is_playing = False
        self.cleanup()

    def cleanup(self):
        self.mutex.lock()
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        finally:
            self.mutex.unlock()

# Stereo waveform timeline view
class StereoWaveformView(QGraphicsView):
    seek = pyqtSignal(float)

    def __init__(self, tracks):
        super().__init__()
        self.tracks = tracks
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setFixedHeight(150)
        self.current_time = 0
        self.time_line = None
        self.mutex = QMutex()

    def update_timeline(self, tracks, current_time):
        self.mutex.lock()
        try:
            self.tracks = tracks
            self.current_time = current_time
            self.scene.clear()
            self.time_line = None

            if not tracks:
                return

            total_duration = sum(track['play_duration'] for track in tracks)
            if total_duration == 0:
                return

            width = self.width()
            pixels_per_second = width / total_duration

            current_x = 0
            for i, track in enumerate(tracks):
                track_width = track['play_duration'] * pixels_per_second
                color = QColor(34, 197, 94, 128) if track['section'] == 'beginning' else \
                        QColor(234, 179, 8, 128) if track['section'] == 'middle' else QColor(239, 68, 68, 128)

                rect = QGraphicsRectItem(current_x, 0, track_width, 100)
                rect.setBrush(QBrush(color))
                self.scene.addItem(rect)

                try:
                    audio = AudioSegment.from_file(track['filepath']).set_channels(2)
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                    samples = samples.reshape(-1, 2)
                    samples_per_pixel = len(samples) / track_width
                    for x in range(0, int(track_width), 2):
                        start_idx = int(x * samples_per_pixel)
                        end_idx = int((x + 2) * samples_per_pixel)
                        if start_idx < len(samples):
                            segment = samples[start_idx:end_idx]
                            left = np.mean(np.abs(segment[:, 0])) * 40
                            right = np.mean(np.abs(segment[:, 1])) * 40
                            self.scene.addItem(QGraphicsLineItem(current_x + x, 50 - left, current_x + x, 50))
                            self.scene.addItem(QGraphicsLineItem(current_x + x, 50, current_x + x, 50 + right))
                except Exception as e:
                    logging.error(f"Error rendering waveform for {track['filepath']}: {e}\n{traceback.format_exc()}")

                text = QGraphicsTextItem(os.path.basename(track['filepath']))
                text.setPos(current_x + 5, 5)
                self.scene.addItem(text)

                start_text = QGraphicsTextItem(self.format_time(track['start_time']))
                start_text.setPos(current_x + 5, 120)
                self.scene.addItem(start_text)
                end_text = QGraphicsTextItem(self.format_time(track['start_time'] + track['play_duration']))
                end_text.setPos(current_x + track_width - 30, 120)
                self.scene.addItem(end_text)

                if 'transition' in track:
                    trans_rect = QGraphicsRectItem(
                        current_x + track_width - track['transition']['duration'] * pixels_per_second,
                        0, track['transition']['duration'] * pixels_per_second, 100
                    )
                    trans_rect.setBrush(QBrush(QColor(59, 130, 246, 178)))
                    self.scene.addItem(trans_rect)
                    trans_text = QGraphicsTextItem(track['transition']['type'])
                    trans_text.setPos(current_x + track_width - track['transition']['duration'] * pixels_per_second + 5, 50)
                    self.scene.addItem(trans_text)

                current_x += track_width

            time_x = (current_time / total_duration) * width
            self.time_line = QGraphicsLineItem(time_x, 0, time_x, 150)
            self.time_line.setPen(QPen(QColor(168, 85, 247), 2))
            self.scene.addItem(self.time_line)

            self.scene.setSceneRect(0, 0, width, 150)
        finally:
            self.mutex.unlock()

    def format_time(self, seconds):
        if not seconds or math.isnan(seconds):
            return '0:00'
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def mousePressEvent(self, event):
        if not self.tracks:
            return
        total_duration = sum(track['play_duration'] for track in self.tracks)
        x = float(event.position().x())
        seek_time = (x / self.width()) * total_duration
        self.seek.emit(seek_time)

# Filter dialog
class FilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Songs")
        layout = QVBoxLayout()

        tempo_layout = QHBoxLayout()
        tempo_layout.addWidget(QLabel("Tempo (BPM):"))
        self.tempo_min = QDoubleSpinBox()
        self.tempo_min.setRange(0, 300)
        self.tempo_min.setValue(0)
        tempo_layout.addWidget(self.tempo_min)
        self.tempo_max = QDoubleSpinBox()
        self.tempo_max.setRange(0, 300)
        self.tempo_max.setValue(300)
        tempo_layout.addWidget(self.tempo_max)
        layout.addLayout(tempo_layout)

        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("Energy:"))
        self.energy_min = QDoubleSpinBox()
        self.energy_min.setRange(0, 10000)
        self.energy_min.setValue(0)
        energy_layout.addWidget(self.energy_min)
        self.energy_max = QDoubleSpinBox()
        self.energy_max.setRange(0, 10000)
        self.energy_max.setValue(10000)
        energy_layout.addWidget(self.energy_max)
        layout.addLayout(energy_layout)

        danceability_layout = QHBoxLayout()
        danceability_layout.addWidget(QLabel("Danceability:"))
        self.danceability_min = QDoubleSpinBox()
        self.danceability_min.setRange(0, 10)
        self.danceability_min.setValue(0)
        danceability_layout.addWidget(self.danceability_min)
        self.danceability_max = QDoubleSpinBox()
        self.danceability_max.setRange(0, 10)
        self.danceability_max.setValue(10)
        danceability_layout.addWidget(self.danceability_max)
        layout.addLayout(danceability_layout)

        apply_button = QPushButton("Apply Filters")
        apply_button.clicked.connect(self.accept)
        layout.addWidget(apply_button)

        self.setLayout(layout)

# Load JSON asynchronously
def load_json_async(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON database async: {e}\n{traceback.format_exc()}")
        return {}

# Database thread
class DatabaseThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, input_dir, audio_files):
        super().__init__()
        self.input_dir = input_dir
        self.audio_files = audio_files
        self.manager = Manager()
        self.database = self.manager.dict()

    def run(self):
        try:
            json_path = os.path.join(self.input_dir, "music_features.json")
            existing_files = set()

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_json_async, json_path)
                loaded_data = future.result()
                self.database.update(loaded_data)
                existing_files = set(loaded_data.keys())
                self.update_signal.emit(f"Loaded partial database with {len(self.database)} songs.")
                logging.info(f"Loaded partial database from {json_path} with {len(self.database)} songs.")

            files_to_analyze = []
            total_files = len(self.audio_files)
            self.update_signal.emit(f"Scanning {total_files} files for updates...")
            logging.info(f"Scanning {total_files} files in directory {self.input_dir}.")

            for i, audio_file in enumerate(self.audio_files):
                try:
                    mtime = os.path.getmtime(audio_file)
                    if audio_file not in self.database or self.database[audio_file].get('last_modified', 0) != mtime:
                        files_to_analyze.append(audio_file)
                    else:
                        logging.info(f"Skipping unchanged file: {audio_file}")
                except Exception as e:
                    self.update_signal.emit(f"Error checking file {audio_file}: {e}")
                    logging.error(f"Error checking file {audio_file}: {e}\n{traceback.format_exc()}")
                    files_to_analyze.append(audio_file)
                self.progress_signal.emit(int((i + 1) / total_files * 50))

            self.update_signal.emit(f"Found {len(files_to_analyze)} new or modified files to analyze.")
            logging.info(f"Found {len(files_to_analyze)} new or modified files to analyze.")

            if files_to_analyze:
                batch_size = int(psutil.cpu_count(logical=True) * 3)
                for i in range(0, len(files_to_analyze), batch_size):
                    batch = files_to_analyze[i:i + batch_size]
                    pool_size, batch_size = get_dynamic_pool_size_and_batch(batch_size, len(files_to_analyze) - i)
                    self.update_signal.emit(f"Processing batch {i // batch_size + 1} ({len(batch)} songs, {pool_size} processes)...")
                    logging.info(f"Processing batch {i // batch_size + 1} ({len(batch)} songs, {pool_size} processes).")
                    
                    start_time = time.time()
                    with ProcessPoolExecutor(max_workers=pool_size) as executor:
                        future_to_file = {executor.submit(extract_features_with_retry, audio_file): audio_file for audio_file in batch}
                        for future in as_completed(future_to_file):
                            audio_file = future_to_file[future]
                            try:
                                features = future.result()
                                if features:
                                    self.database[audio_file] = features
                                    logging.info(f"Indexed features for {audio_file}.")
                                else:
                                    logging.warning(f"Failed to extract features for {audio_file}.")
                            except Exception as e:
                                logging.error(f"Error processing {audio_file}: {e}\n{traceback.format_exc()}")
                    
                    elapsed_time = time.time() - start_time
                    self.update_signal.emit(f"Batch {i // batch_size + 1} completed in {elapsed_time:.2f} seconds.")
                    logging.info(f"Batch {i // batch_size + 1} completed in {elapsed_time:.2f} seconds "
                                f"(CPU: {psutil.cpu_percent():.1f}%, RAM: {psutil.virtual_memory().used / (1024 ** 3):.1f} GB)")
                    
                    self.progress_signal.emit(50 + int((i + len(batch)) / len(files_to_analyze) * 50))
                    gc.collect()

            try:
                with open(json_path, 'w') as f:
                    json.dump(dict(self.database), f, indent=4)
                self.update_signal.emit(f"Completed indexing. Final database saved with {len(self.database)} songs.")
                logging.info(f"Final database saved with {len(self.database)} songs.")
            except Exception as e:
                self.update_signal.emit(f"Error saving final JSON database: {e}")
                logging.error(f"Error saving final JSON database: {e}\n{traceback.format_exc()}")

            self.finished_signal.emit(dict(self.database))
        except Exception as e:
            self.update_signal.emit(f"Critical thread error: {e}")
            logging.error(f"Critical thread error: {e}\n{traceback.format_exc()}")

# Main GUI class
class MusicSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Similarity Search & DJ Mixer")
        self.setGeometry(100, 100, 1200, 800)

        self.input_dir = None
        self.audio_files = []
        self.database = {}
        self.current_playlist = []
        self.current_song_index = -1
        self.current_time = 0
        self.is_playing = False
        self.volume = 0.7
        self.player_thread = None

        self.mix_config = {
            'preset': 'club',
            'mode': 'spokojny-energiczny',
            'duration': 60,
            'beginning_ratio': 30,
            'middle_ratio': 40,
            'ending_ratio': 30,
            'auto_transitions': True,
            'transition_duration': 8,
            'transition_type': 'crossfade',
            'fade_curve': 'linear',
            'harmonic_mixing': True,
            'energy_progression': 'gradual',
            'beat_alignment': True
        }

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)

        self.init_ui()
        self.apply_preset('club')

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)

        left_panel.addWidget(QLabel("Music Directory:"))
        dir_layout = QHBoxLayout()
        self.dir_entry = QLineEdit()
        dir_layout.addWidget(self.dir_entry)
        dir_button = QPushButton("Choose Directory")
        dir_button.clicked.connect(self.choose_directory)
        dir_layout.addWidget(dir_button)
        left_panel.addLayout(dir_layout)

        config_widget = QWidget()
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)
        left_panel.addWidget(config_widget)

        config_layout.addWidget(QLabel("Mix Configuration"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['club', 'chill', 'festival', 'workout'])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        config_layout.addWidget(QLabel("Preset"))
        config_layout.addWidget(self.preset_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            'Spokojny → Energiczny', 'Energiczny → Spokojny', 'Stałe wysokie tempo',
            'Stałe spokojne tempo', 'Stałe energiczne tempo', 'Zbalansowany'
        ])
        config_layout.addWidget(QLabel("Mix Mode"))
        config_layout.addWidget(self.mode_combo)

        self.duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.duration_slider.setRange(15, 120)
        self.duration_slider.setValue(60)
        self.duration_label = QLabel("Duration: 60 min")
        self.duration_slider.valueChanged.connect(self.update_duration)
        config_layout.addWidget(self.duration_label)
        config_layout.addWidget(self.duration_slider)

        ratios_layout = QHBoxLayout()
        self.beginning_spin = QSpinBox()
        self.beginning_spin.setRange(10, 60)
        self.beginning_spin.setValue(30)
        self.middle_spin = QSpinBox()
        self.middle_spin.setRange(10, 60)
        self.middle_spin.setValue(40)
        self.ending_spin = QSpinBox()
        self.ending_spin.setRange(10, 60)
        self.ending_spin.setValue(30)
        ratios_layout.addWidget(QLabel("Beginning"))
        ratios_layout.addWidget(self.beginning_spin)
        ratios_layout.addWidget(QLabel("Middle"))
        ratios_layout.addWidget(self.middle_spin)
        ratios_layout.addWidget(QLabel("Ending"))
        ratios_layout.addWidget(self.ending_spin)
        config_layout.addWidget(QLabel("Section Ratios"))
        config_layout.addLayout(ratios_layout)

        self.transition_type_combo = QComboBox()
        self.transition_type_combo.addItems(['auto', 'crossfade', 'beatmatch-slow', 'beatmatch-medium', 'quick-cut', 'echo', 'high-pass'])
        config_layout.addWidget(QLabel("Transition Type"))
        config_layout.addWidget(self.transition_type_combo)

        self.fade_curve_combo = QComboBox()
        self.fade_curve_combo.addItems(['linear', 'exponential', 'logarithmic'])
        config_layout.addWidget(QLabel("Fade Curve"))
        config_layout.addWidget(self.fade_curve_combo)

        self.transition_duration_slider = QSlider(Qt.Orientation.Horizontal)
        self.transition_duration_slider.setRange(2, 30)
        self.transition_duration_slider.setValue(8)
        self.transition_duration_label = QLabel("Transition Duration: 8s")
        self.transition_duration_slider.valueChanged.connect(self.update_transition_duration)
        config_layout.addWidget(self.transition_duration_label)
        config_layout.addWidget(self.transition_duration_slider)

        self.auto_transitions_check = QCheckBox("Automatic Transitions")
        self.auto_transitions_check.setChecked(True)
        config_layout.addWidget(self.auto_transitions_check)
        self.harmonic_mixing_check = QCheckBox("Harmonic Mixing (Key)")
        self.harmonic_mixing_check.setChecked(True)
        config_layout.addWidget(self.harmonic_mixing_check)
        self.beat_alignment_check = QCheckBox("Beat Alignment")
        self.beat_alignment_check.setChecked(True)
        config_layout.addWidget(self.beat_alignment_check)

        song_layout = QHBoxLayout()
        song_layout.addWidget(QLabel("Select Song:"))
        self.song_combo = QComboBox()
        song_layout.addWidget(self.song_combo)
        song_layout.addWidget(QLabel("Number of Similar Songs:"))
        self.num_similar_spin = QSpinBox()
        self.num_similar_spin.setRange(1, 20)
        self.num_similar_spin.setValue(5)
        song_layout.addWidget(self.num_similar_spin)
        left_panel.addLayout(song_layout)

        analyze_button = QPushButton("Analyze and Find Similar")
        analyze_button.clicked.connect(self.analyze)
        left_panel.addWidget(analyze_button)

        generate_mix_button = QPushButton("Generate Mix")
        generate_mix_button.clicked.connect(self.generate_mix)
        left_panel.addWidget(generate_mix_button)

        export_mix_button = QPushButton("Export Mix")
        export_mix_button.clicked.connect(self.export_mix)
        export_mix_button.setEnabled(False)
        self.export_mix_button = export_mix_button
        left_panel.addWidget(export_mix_button)

        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 2)

        self.timeline = StereoWaveformView(self.current_playlist)
        self.timeline.seek.connect(self.seek_to)
        right_panel.addWidget(self.timeline)

        player_widget = QWidget()
        player_layout = QVBoxLayout()
        player_widget.setLayout(player_layout)
        right_panel.addWidget(player_widget)

        self.track_label = QLabel("Select a track")
        player_layout.addWidget(self.track_label)

        info_layout = QHBoxLayout()
        self.bpm_label = QLabel("BPM: N/A")
        self.key_label = QLabel("Key: N/A")
        self.energy_label = QLabel("Energy: N/A")
        self.transition_label = QLabel("")
        info_layout.addWidget(self.bpm_label)
        info_layout.addWidget(self.key_label)
        info_layout.addWidget(self.energy_label)
        info_layout.addWidget(self.transition_label)
        player_layout.addLayout(info_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        player_layout.addWidget(self.progress_bar)

        self.position_label = QLabel("00:00 / 00:00")
        player_layout.addWidget(self.position_label)

        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_pause)
        playback_layout.addWidget(self.play_button)
        pause_button = QPushButton("Pause")
        pause_button.clicked.connect(self.pause)
        playback_layout.addWidget(pause_button)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self.stop)
        playback_layout.addWidget(stop_button)
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.previous)
        playback_layout.addWidget(prev_button)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_audio)
        playback_layout.addWidget(next_button)
        player_layout.addLayout(playback_layout)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        volume_layout.addWidget(self.volume_slider)
        self.volume_slider.valueChanged.connect(self.update_volume)
        player_layout.addLayout(volume_layout)

        player_layout.addWidget(QLabel("Playlist:"))
        self.playlist_widget = QListWidget()
        player_layout.addWidget(self.playlist_widget)
        self.playlist_widget.itemDoubleClicked.connect(self.play_selected_song)

        playlist_mgmt_layout = QHBoxLayout()
        save_playlist_button = QPushButton("Save Playlist")
        save_playlist_button.clicked.connect(self.save_playlist)
        playlist_mgmt_layout.addWidget(save_playlist_button)
        load_playlist_button = QPushButton("Load Playlist")
        load_playlist_button.clicked.connect(self.load_playlist)
        playlist_mgmt_layout.addWidget(load_playlist_button)
        load_dir_button = QPushButton("Load Directory with Filters")
        load_dir_button.clicked.connect(self.load_directory_with_filters)
        playlist_mgmt_layout.addWidget(load_dir_button)
        player_layout.addLayout(playlist_mgmt_layout)

        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Tempo", "Energy", "Spectral Centroid", "Danceability", "Dynamic Complexity"])
        sort_layout.addWidget(self.sort_combo)
        self.sort_order_combo = QComboBox()
        self.sort_order_combo.addItems(["Ascending", "Descending"])
        sort_layout.addWidget(self.sort_order_combo)
        sort_button = QPushButton("Sort Playlist")
        sort_button.clicked.connect(self.sort_playlist)
        sort_layout.addWidget(sort_button)
        player_layout.addLayout(sort_layout)

        player_layout.addWidget(QLabel("Analysis Results:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        player_layout.addWidget(self.result_text)

    def update_status(self, message):
        self.result_text.append(message)
        self.result_text.ensureCursorVisible()

    def update_progress(self):
        if not self.current_playlist or self.current_song_index >= len(self.current_playlist):
            return
        track = self.current_playlist[self.current_song_index]
        track_duration = track['play_duration']
        progress = (self.current_time - track['start_time']) / track_duration * 100
        self.progress_bar.setValue(int(progress))
        self.position_label.setText(f"{self.format_time(self.current_time - track['start_time'])} / {self.format_time(track_duration)}")
        self.timeline.update_timeline(self.current_playlist, self.current_time)

    def format_time(self, seconds):
        if not seconds or math.isnan(seconds):
            return '0:00'
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def update_duration(self):
        value = self.duration_slider.value()
        self.mix_config['duration'] = value
        self.duration_label.setText(f"Duration: {value} min")

    def update_transition_duration(self):
        value = self.transition_duration_slider.value()
        self.mix_config['transition_duration'] = value
        self.transition_duration_label.setText(f"Transition Duration: {value}s")

    def apply_preset(self, preset_name):
        presets = {
            'club': {
                'preset': 'club',
                'mode': 'spokojny-energiczny',
                'duration': 90,
                'beginning_ratio': 20,
                'middle_ratio': 60,
                'ending_ratio': 10,
                'auto_transitions': True,
                'transition_duration': 8,
                'transition_type': 'crossfade',
                'fade_curve': 'linear',
                'harmonic_mixing': True,
                'energy_progression': 'gradual',
                'beat_alignment': True
            },
            'chill': {
                'preset': 'chill',
                'mode': 'stałe spokojne tempo',
                'duration': 45,
                'beginning_ratio': 30,
                'middle_ratio': 40,
                'ending_ratio': 30,
                'auto_transitions': True,
                'transition_duration': 12,
                'transition_type': 'crossfade',
                'fade_curve': 'exponential',
                'harmonic_mixing': True,
                'energy_progression': 'stable',
                'beat_alignment': False
            },
            'festival': {
                'preset': 'festival',
                'mode': 'stałe energiczne tempo',
                'duration': 120,
                'beginning_ratio': 25,
                'middle_ratio': 50,
                'ending_ratio': 25,
                'auto_transitions': True,
                'transition_duration': 6,
                'transition_type': 'beatmatch-medium',
                'fade_curve': 'linear',
                'harmonic_mixing': True,
                'energy_progression': 'peak',
                'beat_alignment': True
            },
            'workout': {
                'preset': 'workout',
                'mode': 'stałe wysokie tempo',
                'duration': 30,
                'beginning_ratio': 20,
                'middle_ratio': 60,
                'ending_ratio': 20,
                'auto_transitions': True,
                'transition_duration': 4,
                'transition_type': 'quick-cut',
                'fade_curve': 'linear',
                'harmonic_mixing': False,
                'energy_progression': 'high',
                'beat_alignment': True
            }
        }
        self.mix_config.update(presets[preset_name])
        self.preset_combo.setCurrentText(preset_name)
        self.mode_combo.setCurrentText(self.mix_config['mode'])
        self.duration_slider.setValue(self.mix_config['duration'])
        self.duration_label.setText(f"Duration: {self.mix_config['duration']} min")
        self.beginning_spin.setValue(self.mix_config['beginning_ratio'])
        self.middle_spin.setValue(self.mix_config['middle_ratio'])
        self.ending_spin.setValue(self.mix_config['ending_ratio'])
        self.transition_type_combo.setCurrentText(self.mix_config['transition_type'])
        self.fade_curve_combo.setCurrentText(self.mix_config['fade_curve'])
        self.transition_duration_slider.setValue(self.mix_config['transition_duration'])
        self.transition_duration_label.setText(f"Transition Duration: {self.mix_config['transition_duration']}s")
        self.auto_transitions_check.setChecked(self.mix_config['auto_transitions'])
        self.harmonic_mixing_check.setChecked(self.mix_config['harmonic_mixing'])
        self.beat_alignment_check.setChecked(self.mix_config['beat_alignment'])

    def validate_ratios(self):
        total = self.beginning_spin.value() + self.middle_spin.value() + self.ending_spin.value()
        if total != 100:
            self.update_status("Sum of section ratios must be 100%")
            return False
        return True

    def generate_mix(self):
        if not self.database or not self.audio_files:
            self.update_status("Load a directory with tracks first!")
            return

        if not self.validate_ratios():
            return

        try:
            total_duration_sec = self.mix_config['duration'] * 60
            beginning_duration = (total_duration_sec * self.beginning_spin.value()) / 100
            middle_duration = (total_duration_sec * self.middle_spin.value()) / 100
            ending_duration = (total_duration_sec * self.ending_spin.value()) / 100

            tracks = [
                {
                    'filepath': filepath,
                    'filename': os.path.basename(filepath),
                    'name': os.path.basename(filepath).rsplit('.', 1)[0],
                    **self.database[filepath]
                } for filepath in self.audio_files if filepath in self.database
            ]
            sorted_by_energy = sorted(tracks, key=lambda x: calculate_track_energy(x))
            mix = []
            current_time = 0

            beginning_tracks = sorted_by_energy[:int(len(sorted_by_energy) * 0.4)]
            middle_tracks = sorted_by_energy[int(len(sorted_by_energy) * 0.3):]
            ending_tracks = sorted_by_energy[:int(len(sorted_by_energy) * 0.3)] if 'spokojny' in self.mix_config['mode'].lower() else \
                            sorted_by_energy[-int(len(sorted_by_energy) * 0.4):]

            section_tracks = beginning_tracks[:]
            while current_time < beginning_duration and section_tracks:
                track_idx = random.randint(0, len(section_tracks) - 1)
                track = section_tracks[track_idx]
                play_duration = min(track['duration'], beginning_duration - current_time)
                mix_track = {
                    **track,
                    'start_time': current_time,
                    'play_duration': play_duration,
                    'section': 'beginning',
                    'energy': calculate_track_energy(track)
                }
                if mix:
                    mix[-1]['transition'] = generate_transition(mix[-1], track, self.mix_config)
                mix.append(mix_track)
                current_time += play_duration
                section_tracks.pop(track_idx)
                if not section_tracks:
                    section_tracks = beginning_tracks.copy()

            section_tracks = middle_tracks[:]
            middle_start = current_time
            while current_time < middle_start + middle_duration and section_tracks:
                track_idx = random.randint(0, len(section_tracks) - 1)
                track = section_tracks[track_idx]
                play_duration = min(track['duration'], middle_start + middle_duration - current_time)
                mix_track = {
                    **track,
                    'start_time': current_time,
                    'play_duration': play_duration,
                    'section': 'middle',
                    'energy': calculate_track_energy(track)
                }
                if mix:
                    mix[-1]['transition'] = generate_transition(mix[-1], track, self.mix_config)
                mix.append(mix_track)
                current_time += play_duration
                section_tracks.pop(track_idx)
                if not section_tracks:
                    section_tracks = middle_tracks.copy()

            section_tracks = ending_tracks[:]
            while current_time < total_duration_sec and section_tracks:
                track_idx = random.randint(0, len(section_tracks) - 1)
                track = section_tracks[track_idx]
                play_duration = min(track['duration'], total_duration_sec - current_time)
                mix_track = {
                    **track,
                    'start_time': current_time,
                    'play_duration': play_duration,
                    'section': 'ending',
                    'energy': calculate_track_energy(track)
                }
                if mix:
                    mix[-1]['transition'] = generate_transition(mix[-1], track, self.mix_config)
                mix.append(mix_track)
                current_time += play_duration
                section_tracks.pop(track_idx)
                if not section_tracks:
                    section_tracks = ending_tracks.copy()

            self.current_playlist = mix
            self.current_song_index = 0
            self.current_time = 0
            self.is_playing = False
            self.update_playlist_widget()
            self.update_player()
            self.export_mix_button.setEnabled(True)
            self.update_status(f"Generated mix with {len(mix)} tracks.")
            logging.info(f"Generated mix with {len(mix)} tracks.")
        except Exception as e:
            self.update_status(f"Error generating mix: {e}")
            logging.error(f"Error generating mix: {e}\n{traceback.format_exc()}")

    def export_mix(self):
        try:
            mix_data = {
                'config': self.mix_config,
                'tracks': [
                    {
                        'filepath': track['filepath'],
                        'start_time': track['start_time'],
                        'play_duration': track['play_duration'],
                        'section': track['section'],
                        'transition': track.get('transition')
                    } for track in self.current_playlist
                ],
                'total_duration_min': self.mix_config['duration'],
                'created': datetime.now().isoformat()
            }
            json_file_name, _ = QFileDialog.getSaveFileName(self, "Export Mix Metadata", "", "JSON Files (*.json)")
            if json_file_name:
                with open(json_file_name, 'w') as f:
                    json.dump(mix_data, f, indent=2)
                self.update_status(f"Mix metadata exported to {json_file_name}")
                logging.info(f"Mix metadata exported to {json_file_name}")

            audio_file_name, _ = QFileDialog.getSaveFileName(
                self, "Export Audio Mix", "", "WAV Files (*.wav);;MP3 Files (*.mp3)")
            if not audio_file_name:
                self.update_status("Audio export cancelled.")
                return

            if not self.current_playlist:
                self.update_status("No audio files to export!")
                return

            final_mix = AudioSegment.silent(duration=0)
            current_time_ms = 0

            for i, track in enumerate(self.current_playlist):
                try:
                    audio = AudioSegment.from_file(track['filepath']).set_channels(2).set_frame_rate(44100)
                except Exception as e:
                    self.update_status(f"Error loading {track['filename']}: {e}")
                    logging.error(f"Error loading {track['filename']}: {e}\n{traceback.format_exc()}")
                    continue

                play_duration_ms = int(track['play_duration'] * 1000)
                audio = audio[:play_duration_ms]

                if i < len(self.current_playlist) - 1 and 'transition' in track:
                    next_track = self.current_playlist[i + 1]
                    try:
                        next_audio = AudioSegment.from_file(next_track['filepath']).set_channels(2).set_frame_rate(44100)
                    except Exception as e:
                        self.update_status(f"Error loading next track {next_track['filename']}: {e}")
                        logging.error(f"Error loading next track {next_track['filename']}: {e}\n{traceback.format_exc()}")
                        next_audio = AudioSegment.silent(duration=play_duration_ms)

                    transition_duration_ms = int(track['transition']['duration'] * 1000)

                    if track['transition']['type'].startswith('crossfade'):
                        audio = audio.fade_out(transition_duration_ms)
                        next_audio = next_audio[:transition_duration_ms].fade_in(transition_duration_ms)
                        audio = audio.overlay(next_audio, position=play_duration_ms - transition_duration_ms)
                    elif track['transition']['type'] == 'quick-cut':
                        pass
                    elif track['transition']['type'].startswith('beatmatch'):
                        audio = audio.fade_out(transition_duration_ms // 2)
                        next_audio = next_audio[:transition_duration_ms].fade_in(transition_duration_ms // 2)
                        audio = audio.overlay(next_audio, position=play_duration_ms - transition_duration_ms // 2)

                final_mix += audio
                current_time_ms += play_duration_ms

            try:
                final_mix.export(audio_file_name, format=audio_file_name.rsplit('.', 1)[1].lower())
                self.update_status(f"Audio mix exported to {audio_file_name}")
                logging.info(f"Audio mix exported to {audio_file_name}")
            except Exception as e:
                self.update_status(f"Error exporting audio mix: {e}")
                logging.error(f"Error exporting audio mix: {e}\n{traceback.format_exc()}")
        except Exception as e:
            self.update_status(f"Error exporting mix: {e}")
            logging.error(f"Error exporting mix: {e}\n{traceback.format_exc()}")

    def update_player(self):
        try:
            if not self.current_playlist or self.current_song_index >= len(self.current_playlist):
                self.track_label.setText("Select a track")
                self.bpm_label.setText("BPM: N/A")
                self.key_label.setText("Key: N/A")
                self.energy_label.setText("Energy: N/A")
                self.transition_label.setText("")
                return

            track = self.current_playlist[self.current_song_index]
            self.track_label.setText(track.get('name', track['filename']))
            self.bpm_label.setText(f"BPM: {track['tempo']:.1f}")
            self.key_label.setText(f"Key: {track['key']}")
            self.energy_label.setText(f"Energy: {track['energy']:.2f}")
            if 'transition' in track:
                self.transition_label.setText(f"Transition: {track['transition']['type']}")
            else:
                self.transition_label.setText("")
            self.update_progress()
        except Exception as e:
            self.update_status(f"Error updating player: {e}")
            logging.error(f"Error updating player: {e}\n{traceback.format_exc()}")

    def play_pause(self):
        if not self.current_playlist:
            self.update_status("No playlist to play!")
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.player_thread = AudioPlayerThread(self.current_playlist, self.current_song_index, self.volume, self.mix_config)
            self.player_thread.update_time.connect(self.update_current_time)
            self.player_thread.finished.connect(self.next_audio)
            self.player_thread.error.connect(self.handle_audio_error)
            self.player_thread.start()
            self.timer.start(100)
            self.play_button.setText("Pause")
        else:
            if self.player_thread:
                self.player_thread.stop()
            self.timer.stop()
            self.play_button.setText("Play")

    def update_current_time(self, time):
        self.current_time = time
        self.update_progress()

    def handle_audio_error(self, message):
        self.update_status(message)
        self.is_playing = False
        self.play_button.setText("Play")
        self.timer.stop()
        if self.player_thread:
            self.player_thread.stop()

    def pause(self):
        if self.is_playing:
            self.is_playing = False
            if self.player_thread:
                self.player_thread.stop()
            self.timer.stop()
            self.play_button.setText("Play")
            logging.info(f"Paused: {self.current_playlist[self.current_song_index]['filepath']}")

    def stop(self):
        self.is_playing = False
        if self.player_thread:
            self.player_thread.stop()
        self.timer.stop()
        self.current_time = 0
        self.update_progress()
        if self.current_song_index >= 0:
            logging.info(f"Stopped: {self.current_playlist[self.current_song_index]['filepath']}")

    def next_audio(self):
        if self.current_song_index < len(self.current_playlist) - 1:
            self.current_song_index += 1
            self.current_time = self.current_playlist[self.current_song_index]['start_time']
            self.update_playlist_widget()
            self.update_player()
            if self.is_playing:
                if self.player_thread:
                    self.player_thread.stop()
                self.player_thread = AudioPlayerThread(self.current_playlist, self.current_song_index, self.volume, self.mix_config)
                self.player_thread.update_time.connect(self.update_current_time)
                self.player_thread.finished.connect(self.next_audio)
                self.player_thread.error.connect(self.handle_audio_error)
                self.player_thread.start()
                self.timer.start(100)
                logging.info(f"Next song: {self.current_playlist[self.current_song_index]['filepath']}")

    def previous(self):
        if self.current_song_index > 0:
            self.current_song_index -= 1
            self.current_time = self.current_playlist[self.current_song_index]['start_time']
            self.update_playlist_widget()
            self.update_player()
            if self.is_playing:
                if self.player_thread:
                    self.player_thread.stop()
                self.player_thread = AudioPlayerThread(self.current_playlist, self.current_song_index, self.volume, self.mix_config)
                self.player_thread.update_time.connect(self.update_current_time)
                self.player_thread.finished.connect(self.next_audio)
                self.player_thread.error.connect(self.handle_audio_error)
                self.player_thread.start()
                self.timer.start(100)
            logging.info(f"Previous song: {self.current_playlist[self.current_song_index]['filepath']}")

    def play_selected_song(self, item):
        index = self.playlist_widget.row(item)
        if 0 <= index < len(self.current_playlist):
            self.current_song_index = index
            self.current_time = self.current_playlist[index]['start_time']
            self.update_playlist_widget()
            self.update_player()
            if self.is_playing:
                if self.player_thread:
                    self.player_thread.stop()
                self.player_thread = AudioPlayerThread(self.current_playlist, index, self.volume, self.mix_config)
                self.player_thread.update_time.connect(self.update_current_time)
                self.player_thread.finished.connect(self.next_audio)
                self.player_thread.error.connect(self.handle_audio_error)
                self.player_thread.start()
                self.timer.start(100)
                logging.info(f"Playing selected song: {self.current_playlist[index]['filepath']}")

    def seek_to(self, seek_time):
        try:
            if not self.current_playlist:
                return
            accumulated_time = 0
            for i, track in enumerate(self.current_playlist):
                if seek_time < accumulated_time + track['play_duration']:
                    self.current_song_index = i
                    self.current_time = track['start_time'] + (seek_time - accumulated_time)
                    self.update_playlist_widget()
                    self.update_player()
                    if self.is_playing:
                        if self.player_thread:
                            self.player_thread.stop()
                        self.player_thread = AudioPlayerThread(self.current_playlist, self.current_song_index, self.volume, self.mix_config)
                        self.player_thread.update_time.connect(self.update_current_time)
                        self.player_thread.finished.connect(self.next_audio)
                        self.player_thread.error.connect(self.handle_audio_error)
                        self.player_thread.start()
                        self.timer.start(100)
                    break
                accumulated_time += track['play_duration']
        except Exception as e:
            self.update_status(f"Error seeking: {e}")
            logging.error(f"Error seeking: {e}\n{traceback.format_exc()}")

    def update_volume(self, value):
        self.volume = value / 100
        if self.player_thread:
            self.player_thread.volume = self.volume

    def update_playlist_widget(self):
        self.playlist_widget.clear()
        for track in self.current_playlist:
            self.playlist_widget.addItem(os.path.basename(track['filepath']))
        if self.current_song_index >= 0:
            self.playlist_widget.setCurrentRow(self.current_song_index)

    def database_loaded(self, database):
        self.database = database
        self.update_status(f"Database ready ({len(self.database)} songs).")
        self.progress_bar.setValue(100)

    def choose_directory(self):
        self.input_dir = QFileDialog.getExistingDirectory(self, "Choose Music Directory")
        if not self.input_dir:
            self.update_status("No directory selected!")
            return

        self.dir_entry.setText(self.input_dir)
        audio_extensions = ('.mp3', '.wav', '.flac')
        self.audio_files = [str(f) for f in Path(self.input_dir).rglob('*') if f.suffix.lower() in audio_extensions]

        if not self.audio_files:
            self.update_status("No audio files found!")
            self.song_combo.clear()
            return

        self.song_combo.clear()
        self.song_combo.addItems([os.path.basename(f) for f in self.audio_files])
        self.update_status(f"Scanning {len(self.audio_files)} files...")
        
        self.thread = DatabaseThread(self.input_dir, self.audio_files)
        self.thread.update_signal.connect(self.update_status)
        self.thread.progress_signal.connect(self.update_progress_bar)
        self.thread.finished_signal.connect(self.database_loaded)
        self.thread.start()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def sort_playlist(self):
        if not self.current_playlist or not self.database:
            self.update_status("No playlist or database to sort!")
            return

        sort_key = self.sort_combo.currentText().lower().replace(" ", "_")
        reverse = self.sort_order_combo.currentText() == "Descending"

        key_mapping = {
            "tempo": "tempo",
            "energy": "energy_score",
            "spectral_centroid": "spectral_centroid",
            "danceability": "danceability",
            "dynamic_complexity": "dynamic_complexity"
        }

        sort_field = key_mapping.get(sort_key)
        if not sort_field:
            self.update_status("Invalid sort criterion!")
            return

        try:
            sorted_playlist = sorted(
                self.current_playlist,
                key=lambda x: x.get(sort_field, 0),
                reverse=reverse
            )
            self.current_playlist = sorted_playlist
            self.current_song_index = 0
            self.current_time = self.current_playlist[0]['start_time'] if self.current_playlist else 0
            self.update_playlist_widget()
            self.update_player()
            self.update_status(f"Playlist sorted by {sort_key} ({'descending' if reverse else 'ascending'}).")
            logging.info(f"Playlist sorted by {sort_key} ({'descending' if reverse else 'ascending'}).")
        except Exception as e:
            self.update_status(f"Error sorting playlist: {e}")
            logging.error(f"Error sorting playlist: {e}\n{traceback.format_exc()}")

    def load_directory_with_filters(self):
        if not self.input_dir or not self.database:
            self.update_status("Choose a directory and load database first!")
            return

        dialog = FilterDialog(self)
        if dialog.exec():
            tempo_min = dialog.tempo_min.value()
            tempo_max = dialog.tempo_max.value()
            energy_min = dialog.energy_min.value()
            energy_max = dialog.energy_max.value()
            danceability_min = dialog.danceability_min.value()
            danceability_max = dialog.danceability_max.value()

            filtered_songs = []
            for audio_file, features in self.database.items():
                if (tempo_min <= features.get('tempo', 0) <= tempo_max and
                    energy_min <= features.get('energy', 0) <= energy_max and
                    danceability_min <= features.get('danceability', 0) <= danceability_max):
                    filtered_songs.append({
                        'filepath': audio_file,
                        'filename': os.path.basename(audio_file),
                        'name': os.path.basename(audio_file).rsplit('.', 1)[0],
                        **features,
                        'start_time': 0,
                        'play_duration': features['duration'],
                        'section': 'middle',
                        'energy': calculate_track_energy(features)
                    })

            if not filtered_songs:
                self.update_status("No songs match the filter criteria!")
                return

            self.current_playlist = filtered_songs
            self.current_song_index = 0
            self.current_time = 0
            self.update_playlist_widget()
            self.update_player()
            self.update_status(f"Loaded {len(filtered_songs)} songs after filtering.")
            logging.info(f"Loaded {len(filtered_songs)} songs after filtering: "
                         f"Tempo [{tempo_min}, {tempo_max}], Energy [{energy_min}, {energy_max}], "
                         f"Danceability [{danceability_min}, {danceability_max}]")
            if self.is_playing:
                self.play_pause()

    def save_playlist(self):
        if not self.current_playlist:
            return

        playlist_data = []
        for track in self.current_playlist:
            playlist_data.append({
                'path': track['path'],
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
            })
        try:
            playlist_path = os.path.join(self.input_dir, "playlist.json")
            with open(playlist_path, 'w') as f:
                json.dump(playlist_data, f, indent=4)
            self.update_status(f"Saved playlist to {playlist_path}")
            logging.info(f"Saved playlist: {playlist_path}")
        except Exception as e:
            self.update_status(f"Error saving playlist: {e}")
            logging.error(f"Error saving playlist: {error}\n{traceback.format_exc()}")

    def load_playlist(self):
        if not self.input_dir:
            self.update_status("No input directory selected!")
            return
        playlist_path = os.path.join(self.input_dir, "playlist.json")
        try:
            with open(playlist_path, 'r') as f:
                playlist_data = json.load(f)
            songs = []
            for data in playlist_data:
                audio_file = data.get('path')
                if audio_file in self.database:
                    songs.append({
                        'filepath': audio_file,
                        'filename': os.path.basename(audio_file),
                        'name': os.path.basename(audio_file).rsplit('.', 1)[0],
                        **self.database[audio_file],
                        'start_time': 0,
                        'play_duration': self.database[audio_file]['duration'],
                        'section': 'song',
                        'energy': calculate_track_energy(self.database[audio_file])
                    })
            if not songs:
                self.update_status("No valid songs found in playlist!")
                logging.info(f"No valid songs found in playlist: {playlist_path}")
                return
            self.current_playlist = songs
            self.current_song_index = 0
            self.current_time = 0
            self.update_playlist_widget()
            self.update_player()
            self.update_status(f"Loaded playlist with {len(songs)} songs.")
            logging.info(f"Loaded playlist {playlist_path} with {len(songs)} songs.")
        except Exception as e:
            self.update_status(f"Error loading playlist: {e}")
            logging.error(f"Error loading playlist: {e}\n{traceback.format_exc()}")

    def analyze(self):
        if not self.database or not self.audio_files:
            self.update_status("Load a directory with tracks first!")
            return

        selected_song = self.song_combo.currentText()
        if not selected_song:
            self.update_status("No song selected!")
            return

        selected_file = next((f for f in self.audio_files if os.path.basename(f) == selected_song), None)
        if not selected_file or selected_file not in self.database:
            self.update_status("Selected song not in database!")
            return

        num_songs = self.num_similar_spin.value()
        selected_features = self.database[selected_file]
        similarities = []
        for filepath, features in self.database.items():
            if filepath != selected_file:
                distance = calculate_similarity(selected_features, features)
                similarities.append((distance, filepath))

        similarities.sort(key=lambda x: x[0])
        similar_songs = similarities[:num_songs]

        self.current_playlist = [
            {
                'filepath': selected_file,
                'filename': os.path.basename(selected_file),
                'name': os.path.basename(selected_file).rsplit('.', 1)[0],
                **self.database[selected_file],
                'start_time': 0,
                'play_duration': self.database[selected_file]['duration'],
                'section': 'middle',
                'energy': calculate_track_energy(self.database[selected_file])
            }
        ]

        for distance, filepath in similar_songs:
            self.current_playlist.append({
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'name': os.path.basename(filepath).rsplit('.', 1)[0],
                **self.database[filepath],
                'start_time': 0,
                'play_duration': self.database[filepath]['duration'],
                'section': 'middle',
                'energy': calculate_track_energy(self.database[filepath])
            })

        self.current_song_index = 0
        self.current_time = 0
        self.update_playlist_widget()
        self.update_player()
        self.update_status(f"Found {len(similar_songs)} songs similar to {selected_song}")
        logging.info(f"Found {len(similar_songs)} songs similar to {selected_file}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicSimilarityApp()
    window.show()
    sys.exit(app.exec())
