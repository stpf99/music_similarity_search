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
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QComboBox, QTextEdit, QFileDialog, QProgressBar,
                             QSpinBox, QListWidget, QSlider, QDialog, QDoubleSpinBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from multiprocessing import Pool, Manager
import pygame
import logging
import gc
import traceback

# Increase recursion limit as a precaution
sys.setrecursionlimit(2000)

# Configure logging
logging.basicConfig(
    filename='music_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize pygame mixer
pygame.mixer.init()

# Function to extract audio features from a file
def extract_features(audio_file):
    try:
        loader = es.MonoLoader(filename=audio_file)
        audio = loader()

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
            'similarity_results': []
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

        features['tempo'], _, _, _, _ = tempo(audio)
        key, scale, strength = key(audio)
        features['key'] = f"{key} {scale}"
        features['danceability'] = danceability(audio)[0]
        features['dynamic_complexity'] = dynamic_complexity(audio)[0]
        features['last_modified'] = os.path.getmtime(audio_file)

        del audio
        gc.collect()
        return features
    except Exception as e:
        logging.error(f"Error analyzing {audio_file}: {e}\n{traceback.format_exc()}")
        return None

# Function to calculate similarity between songs
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

# Function to dynamically adjust batch size based on system resources
def get_dynamic_batch_size(current_batch_size, total_files):
    max_cpu_percent = 50
    max_memory = 11  # 70% of 16 GB
    min_batch_size = max(1, psutil.cpu_count(logical=False) // 2)  # Use half of physical cores
    max_batch_size = psutil.cpu_count(logical=False) * 2  # Max batch size based on cores

    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_used = psutil.virtual_memory().used / (1024 ** 3)
    memory_per_file = 0.2

    if cpu_percent > max_cpu_percent * 0.9 or memory_used > max_memory * 0.9:
        new_batch_size = max(min_batch_size, int(current_batch_size * 0.75))
    elif cpu_percent < max_cpu_percent * 0.5 and memory_used < max_memory * 0.5:
        new_batch_size = min(max_batch_size, int(current_batch_size * 1.25))
    else:
        new_batch_size = current_batch_size

    new_batch_size = min(new_batch_size, total_files)
    logging.info(f"Adjusted batch size: {new_batch_size} (CPU: {cpu_percent:.1f}%, RAM: {memory_used:.1f} GB, Cores: {psutil.cpu_count(logical=False)})")
    return new_batch_size

# Thread class for loading database with multiprocessing
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
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        self.database.update(json.load(f))
                    self.update_signal.emit(f"Loaded existing database ({len(self.database)} songs).")
                    logging.info(f"Loaded database from {json_path}, {len(self.database)} songs.")
                except Exception as e:
                    self.update_signal.emit(f"Error loading JSON database: {e}")
                    logging.error(f"Error loading JSON database: {e}\n{traceback.format_exc()}")

            files_to_analyze = []
            total_files = len(self.audio_files)
            self.update_signal.emit(f"Scanning {total_files} files...")
            logging.info(f"Scanning {total_files} files in directory {self.input_dir}.")
            for i, audio_file in enumerate(self.audio_files):
                mtime = os.path.getmtime(audio_file)
                if audio_file not in self.database or self.database[audio_file]['last_modified'] != mtime:
                    files_to_analyze.append(audio_file)
                self.progress_signal.emit(int((i + 1) / total_files * 50))
                if i % 100 == 0:
                    QApplication.processEvents()

            self.update_signal.emit(f"Found {len(files_to_analyze)} new/changed files to analyze.")
            logging.info(f"Found {len(files_to_analyze)} new/changed files to analyze.")

            if files_to_analyze:
                batch_size = get_dynamic_batch_size(10, len(files_to_analyze))
                with Pool(processes=psutil.cpu_count(logical=False)) as pool:
                    for i in range(0, len(files_to_analyze), batch_size):
                        batch = files_to_analyze[i:i + batch_size]
                        self.update_signal.emit(f"Processing batch {i // batch_size + 1} ({len(batch)} songs)...")
                        logging.info(f"Processing batch {i // batch_size + 1} ({len(batch)} songs).")
                        results = pool.map(extract_features, batch)
                        for audio_file, features in zip(batch, results):
                            if features:
                                self.database[audio_file] = features
                        self.progress_signal.emit(50 + int((i + len(batch)) / len(files_to_analyze) * 50))
                        if len(batch) % 5 == 0:
                            QApplication.processEvents()

                        try:
                            with open(json_path, 'w') as f:
                                json.dump(dict(self.database), f, indent=4)
                            self.update_signal.emit(f"Saved database after batch {i // batch_size + 1}.")
                            logging.info(f"Saved database after batch {i // batch_size + 1}.")
                        except Exception as e:
                            self.update_signal.emit(f"Error saving JSON database: {e}")
                            logging.error(f"Error saving JSON database: {e}\n{traceback.format_exc()}")

            self.finished_signal.emit(dict(self.database))
        except Exception as e:
            self.update_signal.emit(f"Critical thread error: {e}")
            logging.error(f"Critical thread error: {e}\n{traceback.format_exc()}")

# Filter dialog for loading directory with filters
class FilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Songs")
        layout = QVBoxLayout()

        # Tempo filter
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

        # Energy filter
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

        # Danceability filter
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

        # Apply button
        apply_button = QPushButton("Apply Filters")
        apply_button.clicked.connect(self.accept)
        layout.addWidget(apply_button)

        self.setLayout(layout)

# PyQt6 GUI class
class MusicSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Similarity Search")
        self.setGeometry(100, 100, 800, 600)

        self.input_dir = None
        self.audio_files = []
        self.database = {}
        self.current_playlist = []
        self.current_song_index = -1
        self.current_song_duration = 0
        self.current_position = 0

        # Timer for updating playback position
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)
        self.position_timer.setInterval(1000)  # Update every second

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Directory selection
        main_layout.addWidget(QLabel("Music Directory:"))
        dir_layout = QHBoxLayout()
        self.dir_entry = QLineEdit()
        dir_layout.addWidget(self.dir_entry)
        dir_button = QPushButton("Choose Directory")
        dir_button.clicked.connect(self.choose_directory)
        dir_layout.addWidget(dir_button)
        main_layout.addLayout(dir_layout)

        # Song selection and number of similar songs
        song_layout = QHBoxLayout()
        song_layout.addWidget(QLabel("Select Song:"))
        self.song_combo = QComboBox()
        song_layout.addWidget(self.song_combo)
        song_layout.addWidget(QLabel("Number of Similar Songs:"))
        self.num_similar_spin = QSpinBox()
        self.num_similar_spin.setRange(1, 20)
        self.num_similar_spin.setValue(5)
        song_layout.addWidget(self.num_similar_spin)
        main_layout.addLayout(song_layout)

        # Analyze button
        analyze_button = QPushButton("Analyze and Find Similar")
        analyze_button.clicked.connect(self.analyze)
        main_layout.addWidget(analyze_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Playlist
        main_layout.addWidget(QLabel("Playlist:"))
        self.playlist_widget = QListWidget()
        self.playlist_widget.itemDoubleClicked.connect(self.play_selected_song)
        main_layout.addWidget(self.playlist_widget)

        # Playback controls
        playback_layout = QHBoxLayout()
        play_button = QPushButton("Play")
        play_button.clicked.connect(self.play)
        playback_layout.addWidget(play_button)
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
        next_button.clicked.connect(self.next)
        playback_layout.addWidget(next_button)
        main_layout.addLayout(playback_layout)

        # Playback position slider
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)
        main_layout.addWidget(self.position_slider)

        # Playback position label
        self.position_label = QLabel("00:00 / 00:00")
        main_layout.addWidget(self.position_label)

        # Playlist management
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
        main_layout.addLayout(playlist_mgmt_layout)

        # Sorting controls
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
        main_layout.addLayout(sort_layout)

        # Results display
        main_layout.addWidget(QLabel("Analysis Results:"))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        main_layout.addWidget(self.result_text)

    def update_status(self, message):
        self.result_text.append(message)
        self.result_text.ensureCursorVisible()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_position(self):
        if pygame.mixer.music.get_busy():
            self.current_position += 1
            self.position_slider.setValue(self.current_position)
            self.update_position_label()
            if self.current_position >= self.current_duration:
                self.next()

    def update_position_label(self):
        pos_sec = self.current_position
        dur_sec = self.current_duration
        pos_str = f"{pos_sec // 60:02}:{pos_sec % 60:02}"
        dur_str = f"{dur_sec // 60:02}:{dur_sec % 60:02}"
        self.position_label.setText(f"{pos_str} / {dur_str}")

    def set_position(self, position):
        self.current_position = position
        pygame.mixer.music.play(start=position)
        self.position_timer.start()
        self.update_position_label()

    def play(self):
        if self.current_song_index >= 0 and self.current_song_index < len(self.current_playlist):
            if not pygame.mixer.music.get_busy():
                self.play_song()
        else:
            self.update_status("No song to play.")

    def pause(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.position_timer.stop()
            logging.info(f"Paused: {self.current_playlist[self.current_song_index]}")

    def stop(self):
        pygame.mixer.music.stop()
        self.position_timer.stop()
        self.current_position = 0
        self.position_slider.setValue(0)
        self.update_position_label()
        if self.current_song_index >= 0:
            logging.info(f"Stopped: {self.current_playlist[self.current_song_index]}")

    def next(self):
        if self.current_song_index < len(self.current_playlist) - 1:
            self.current_song_index += 1
            self.play_song()
            logging.info(f"Next song: {self.current_playlist[self.current_song_index]}")

    def previous(self):
        if self.current_song_index > 0:
            self.current_song_index -= 1
            self.play_song()
            logging.info(f"Previous song: {self.current_playlist[self.current_song_index]}")

    def play_selected_song(self, item):
        index = self.playlist_widget.row(item)
        if 0 <= index < len(self.current_playlist):
            self.current_song_index = index
            self.play_song()
            logging.info(f"Playing selected song: {self.current_playlist[self.current_song_index]}")

    def play_song(self):
        if 0 <= self.current_song_index < len(self.current_playlist):
            song_path = self.current_playlist[self.current_song_index]
            try:
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
                self.position_timer.start()
                self.current_position = 0
                # Estimate duration
                self.current_duration = self.estimate_duration(song_path)
                self.position_slider.setRange(0, self.current_duration)
                self.update_position_label()
                self.playlist_widget.setCurrentRow(self.current_song_index)
                self.update_status(f"Playing: {os.path.basename(song_path)}")
            except Exception as e:
                self.update_status(f"Playback error: {e}")
                logging.error(f"Playback error {song_path}: {e}\n{traceback.format_exc()}")

    def estimate_duration(self, song_path):
        try:
            loader = es.MonoLoader(filename=song_path)
            audio = loader()
            duration = len(audio) / 44100  # Assuming 44.1kHz sample rate
            del audio
            return int(duration)
        except:
            return 180  # Default 3 minutes if estimation fails

    def save_playlist(self):
        if not self.current_playlist:
            self.update_status("No playlist to save!")
            return
        playlist_data = {
            "songs": self.current_playlist,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            playlist_path = os.path.join(self.input_dir, "playlist.json")
            with open(playlist_path, 'w') as f:
                json.dump(playlist_data, f, indent=4)
            self.update_status(f"Saved playlist to {playlist_path}.")
            logging.info(f"Saved playlist: {playlist_path}")
        except Exception as e:
            self.update_status(f"Error saving playlist: {e}")
            logging.error(f"Error saving playlist: {e}\n{traceback.format_exc()}")

    def load_playlist(self):
        if not self.input_dir:
            self.update_status("Choose a directory first!")
            return
        playlist_path = os.path.join(self.input_dir, "playlist.json")
        if not os.path.exists(playlist_path):
            self.update_status("No saved playlist found!")
            return
        try:
            with open(playlist_path, 'r') as f:
                playlist_data = json.load(f)
            self.current_playlist = playlist_data.get("songs", [])
            self.update_playlist_widget()
            self.update_status(f"Loaded playlist from {playlist_path}.")
            logging.info(f"Loaded playlist: {playlist_path}")
            self.current_song_index = 0
            if self.current_playlist:
                self.play_song()
        except Exception as e:
            self.update_status(f"Error loading playlist: {e}")
            logging.error(f"Error loading playlist: {e}\n{traceback.format_exc()}")

    def update_playlist_widget(self):
        self.playlist_widget.clear()
        for song in self.current_playlist:
            self.playlist_widget.addItem(os.path.basename(song))
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
            self.update_status("No audio files in directory!")
            self.song_combo.clear()
            return

        self.song_combo.clear()
        self.song_combo.addItems([os.path.basename(f) for f in self.audio_files])
        self.update_status(f"Scanning directory ({len(self.audio_files)} files)...")

        self.thread = DatabaseThread(self.input_dir, self.audio_files)
        self.thread.update_signal.connect(self.update_status)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.database_loaded)
        self.thread.start()

    def sort_playlist(self):
        if not self.current_playlist or not self.database:
            self.update_status("No playlist or database to sort!")
            return

        sort_key = self.sort_combo.currentText().lower().replace(" ", "_")
        reverse = self.sort_order_combo.currentText() == "Descending"

        key_mapping = {
            "tempo": "tempo",
            "energy": "energy",
            "spectral_centroid": "spectral_centroid",
            "danceability": "danceability",
            "dynamic_complexity": "dynamic_complexity"
        }

        sort_field = key_mapping.get(sort_key)
        if not sort_field:
            self.update_status("Invalid sorting criterion!")
            return

        try:
            sorted_playlist = sorted(
                self.current_playlist,
                key=lambda x: self.database.get(x, {}).get(sort_field, 0),
                reverse=reverse
            )
            self.current_playlist = sorted_playlist
            self.current_song_index = 0
            self.update_playlist_widget()
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
                    filtered_songs.append(audio_file)

            if not filtered_songs:
                self.update_status("No songs match the filter criteria!")
                return

            self.current_playlist = filtered_songs
            self.current_song_index = 0
            self.update_playlist_widget()
            self.update_status(f"Loaded {len(self.current_playlist)} songs after filtering.")
            logging.info(f"Loaded {len(self.current_playlist)} songs after filtering: "
                         f"Tempo [{tempo_min}, {tempo_max}], Energy [{energy_min}, {energy_max}], "
                         f"Danceability [{danceability_min}, {danceability_max}]")
            if self.current_playlist:
                self.play_song()

    def analyze(self):
        if not self.audio_files or not self.input_dir:
            self.update_status("Choose a directory first!")
            return

        selected_idx = self.song_combo.currentIndex()
        selected_file = self.audio_files[selected_idx]
        self.update_status(f"\nSelected song: {os.path.basename(selected_file)}")

        num_similar = self.num_similar_spin.value()
        selected_features = self.database.get(selected_file)

        if not selected_features:
            self.update_status("Analyzing selected song...")
            selected_features = extract_features(selected_file)
            if selected_features:
                self.database[selected_file] = selected_features
                try:
                    with open(os.path.join(self.input_dir, "music_features.json"), 'w') as f:
                        json.dump(self.database, f, indent=4)
                    self.update_status("Saved song features to database.")
                    logging.info(f"Saved features for: {selected_file} to database.")
                except Exception as e:
                    self.update_status(f"Error saving JSON database: {e}")
                    logging.error(f"Error saving JSON database for {selected_file}: {e}\n{traceback.format_exc()}")
            else:
                self.update_status("Failed to analyze song!")
                return

        # Check for existing similarity results
        if selected_features.get('similarity_results') and len(selected_features['similarity_results']) >= num_similar:
            self.update_status("Loaded similarity results from database.")
            similarities = selected_features['similarity_results'][:num_similar]
            logging.info(f"Loaded saved similarity results for {selected_file} (num_similar: {num_similar}).")
        else:
            self.update_status("Calculating similarity...")
            similarities = []
            for audio_file, features in self.database.items():
                if audio_file == selected_file:
                    continue
                distance = calculate_similarity(selected_features, features)
                similarities.append((audio_file, distance, features))

            similarities.sort(key=lambda x: x[1])
            similarities = similarities[:num_similar]

            # Save similarity results to database
            selected_features['similarity_results'] = [
                {
                    "file": audio_file,
                    "distance": float(distance),
                    "features": {
                        "tempo": features['tempo'],
                        "key": features['key'],
                        "spectral_centroid": features['spectral_centroid'],
                        "energy": features['energy'],
                        "danceability": features['danceability'],
                        "dynamic_complexity": features['dynamic_complexity']
                    }  # Added closing brace
                }
                for audio_file, distance, features in similarities
            ]
            self.database[selected_file] = selected_features
            try:
                with open(os.path.join(self.input_dir, "music_features.json"), 'w') as f:
                    json.dump(self.database, f, indent=4)
                self.update_status(f"Saved similarity results to database ({num_similar} songs).")
                logging.info(f"Saved similarity results for {selected_file}: {num_similar} songs.")
            except Exception as e:
                self.update_status(f"Error saving results for {selected_file}: {e}")
                logging.error(f"Error saving results for {selected_file}: {e}\n{traceback.format_exc()}")

        # Create playlist
        self.current_playlist = [selected_file] + [item['file'] if isinstance(item, dict) else item[0] for item in similarities]
        self.current_song_index = 0
        self.update_playlist_widget()
        self.update_status(f"Created playlist with {len(self.current_playlist)} songs.")
        logging.info(f"Created playlist for {selected_file} with {len(self.current_playlist)} songs.")

        # Display results
        self.result_text.clear()
        self.result_text.append(f"Selected song: {os.path.basename(selected_file)}\n\n")
        self.result_text.append(f"Most similar songs ({num_similar}):\n")
        for i, item in enumerate(similarities, 1):
            audio_file = item['file'] if isinstance(item, dict) else item[0]
            distance = item['distance'] if isinstance(item, dict) else item[1]
            features = item['features'] if isinstance(item, dict) else item[2]
            self.result_text.append(f"{i}. {os.path.basename(audio_file)}")
            self.result_text.append(f"   Distance: {distance:.2f}")
            self.result_text.append(f"   Tempo: {features['tempo']:.1f} BPM")
            self.result_text.append(f"   Key: {features['key']}")
            self.result_text.append(f"   Energy: {features['energy']:.2f}")
            self.result_text.append(f"   Spectral Centroid: {features['spectral_centroid']:.2f} Hz")
            self.result_text.append(f"   Danceability: {features['danceability']:.2f}")
            self.result_text.append(f"   Dynamic Complexity: {features['dynamic_complexity']:.2f}\n")

            logging.info(f"Similarity result {i} for {selected_file}: {os.path.basename(audio_file)}, "
                         f"Distance: {distance:.2f}, Tempo: {features['tempo']:.1f} BPM, "
                         f"Key: {features['key']}, Energy: {features['energy']:.2f}, "
                         f"Centroid: {features['spectral_centroid']:.2f} Hz, "
                         f"Danceability: {features['danceability']:.2f}, "
                         f"Dynamic Complexity: {features['dynamic_complexity']:.2f}")

# Main function
def main():
    app = QApplication(sys.argv)
    window = MusicSimilarityApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
