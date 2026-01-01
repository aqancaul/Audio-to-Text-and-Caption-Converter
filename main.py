#!/usr/bin/env python3
"""
Aplikasi Auto Captioning
Mendukung captioning file video dan capture audio sistem secara real-time
"""

import sys
import os
import platform
import threading
import queue
import json
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QComboBox, QSpinBox, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QTextCursor

import whisper
import sounddevice as sd
import numpy as np
from moviepy import VideoFileClip

# Import khusus platform
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

if IS_LINUX:
    try:
        import pulsectl
        PULSE_AVAILABLE = True
    except ImportError:
        PULSE_AVAILABLE = False
else:
    PULSE_AVAILABLE = False

if IS_WINDOWS:
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        PYCAW_AVAILABLE = True
    except ImportError:
        PYCAW_AVAILABLE = False
else:
    PYCAW_AVAILABLE = False


class CaptionWorker(QObject):
    """Kelas worker untuk memproses audio dan menghasilkan caption"""
    caption_ready = pyqtSignal(str, float, float)  # teks, waktu_mulai, waktu_selesai
    progress_update = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_name="base", language="en"):
        super().__init__()
        self.model = None
        self.model_name = model_name
        self.language = language
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # Proses chunk 3 detik

    def load_model(self):
        """Memuat model Whisper"""
        try:
            self.model = whisper.load_model(self.model_name)
        except Exception as e:
            self.error.emit(f"Error loading model: {str(e)}")

    def process_audio_file(self, audio_path):
        """Memproses file audio dan menghasilkan caption"""
        if not self.model:
            self.load_model()
        
        if self.model is None:
            self.error.emit("Failed to load Whisper model")
            return
        
        self.is_running = True
        try:
            # Validasi file audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            print(f"Processing audio file: {audio_path} ({file_size} bytes)")
            
            # Load dan validasi audio menggunakan whisper
            audio = whisper.load_audio(audio_path)
            
            # Cek apakah audio valid (tidak semua NaN atau zero)
            if len(audio) == 0:
                raise ValueError("Audio file contains no data")
            
            if np.all(np.isnan(audio)) or np.all(audio == 0):
                raise ValueError("Audio file contains invalid data (all NaN or zeros)")
            
            print(f"Audio loaded: {len(audio)} samples, sample rate: {whisper.audio.SAMPLE_RATE} Hz")
            
            # Transcribe dengan validasi tambahan
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                verbose=False,
                fp16=False  # Gunakan fp32 untuk menghindari NaN issues
            )
            
            if not result or "segments" not in result:
                raise ValueError("Whisper returned empty result")
            
            if not result["segments"]:
                self.error.emit("No speech detected in audio file")
                return
            
            # Emit segments
            total_duration = result["segments"][-1]["end"] if result["segments"] else 1.0
            for segment in result["segments"]:
                if not self.is_running:
                    break
                text = segment.get("text", "").strip()
                if text:
                    self.caption_ready.emit(
                        text,
                        segment.get("start", 0.0),
                        segment.get("end", 0.0)
                    )
                    # Update progress
                    progress = int((segment.get("end", 0.0) / total_duration) * 100)
                    self.progress_update.emit(progress)
            
            self.finished.emit()
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)
        finally:
            self.is_running = False

    def process_realtime_audio(self, audio_data):
        """Memproses chunk audio real-time"""
        if not self.model:
            self.load_model()
        
        if self.model is None:
            return
        
        try:
            # Audio data sudah dalam format int16 dari callback
            # Konversi ke float32 dan normalisasi ke range [-1.0, 1.0]
            # int16 range: [-32768, 32767], normalisasi dengan 32768.0
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Pastikan audio tidak kosong
            if len(audio_float) == 0:
                return
            
            # Whisper expects float32 audio in range [-1.0, 1.0]
            result = self.model.transcribe(
                audio_float,
                language=self.language,
                task="transcribe",
                verbose=False,
                fp16=False
            )
            
            if result and "segments" in result and result["segments"]:
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        start_time = segment.get("start", 0.0)
                        end_time = segment.get("end", 0.0)
                        self.caption_ready.emit(text, start_time, end_time)
        except Exception as e:
            error_msg = f"Error processing real-time audio: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

    def stop(self):
        """Menghentikan proses"""
        self.is_running = False


class AudioCapture:
    """Menangani capture audio sistem - Dukungan cross-platform"""
    def __init__(self, sample_rate=16000, channels=1, chunk_size=48000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_recording = False
        self.audio_stream = None
        self.pulse = None
        self.callback = None
        self.platform = platform.system()

    def _find_linux_monitor_device(self):
        """Mencari device monitor di Linux (PipeWire/PulseAudio)"""
        device_id = None
        monitor_source_name = None
        monitor_source_index = None
        
        # Coba PulseAudio/PipeWire via pulsectl jika tersedia
        if PULSE_AVAILABLE:
            try:
                self.pulse = pulsectl.Pulse('audio-capture')
                sources = self.pulse.source_list()
                
                # Prioritaskan mencari monitor dari default sink
                try:
                    default_sink = self.pulse.server_info().default_sink_name
                    # Monitor biasanya bernama: <sink_name>.monitor
                    expected_monitor_name = f"{default_sink}.monitor"
                    
                    for source in sources:
                        # Prioritas 1: Monitor dari default sink
                        if source.name == expected_monitor_name:
                            monitor_source_name = source.name
                            monitor_source_index = source.index
                            break
                        # Prioritas 2: Monitor lainnya
                        elif source.name.endswith('.monitor') and monitor_source_name is None:
                            monitor_source_name = source.name
                            monitor_source_index = source.index
                except Exception as e:
                    print(f"Error getting default sink: {str(e)}")
                    # Fallback: cari monitor apapun
                    for source in sources:
                        if source.name.endswith('.monitor'):
                            monitor_source_name = source.name
                            monitor_source_index = source.index
                            break
                
                print(f"Found monitor source: {monitor_source_name} (index: {monitor_source_index})")
                
                # Set monitor source sebagai default source sementara
                # Ini memastikan sounddevice menggunakan monitor saat membuka stream
                try:
                    # Simpan default source saat ini untuk restore nanti
                    try:
                        old_default = self.pulse.server_info().default_source_name
                        self._old_default_source = old_default
                        print(f"Saved current default source: {old_default}")
                    except:
                        self._old_default_source = None
                    
                    # Set monitor sebagai default source
                    self.pulse.source_default_set(monitor_source_name)
                    print(f"Set monitor source as default: {monitor_source_name}")
                except Exception as e:
                    print(f"Warning: Could not set monitor as default source: {str(e)}")
                    print("Will try to use PipeWire/Pulse device directly...")
                
                # Cari device yang cocok di sounddevice
                if monitor_source_name:
                    devices = sd.query_devices()
                    print(f"\nAvailable sounddevice input devices:")
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            print(f"  [{i}] {device['name']} (channels: {device['max_input_channels']})")
                    
                    # Untuk PipeWire/PulseAudio, jika monitor source ditemukan via pulsectl,
                    # gunakan device PipeWire/Pulse secara langsung karena monitor sources
                    # di-routing melalui device tersebut
                    
                    # Prioritas 1: Cari device PipeWire (untuk PipeWire)
                    for i, device in enumerate(devices):
                        if device['max_input_channels'] > 0:
                            device_name_lower = device['name'].lower()
                            if device_name_lower == 'pipewire' or 'pipewire' in device_name_lower:
                                device_id = i
                                print(f"\n✓ Using PipeWire device [{i}] {device['name']} for monitor source")
                                print(f"  Monitor source will be routed through this device")
                                break
                    
                    # Prioritas 2: Jika tidak ada PipeWire, coba PulseAudio
                    if device_id is None:
                        for i, device in enumerate(devices):
                            if device['max_input_channels'] > 0:
                                device_name_lower = device['name'].lower()
                                if device_name_lower == 'pulse' or ('pulse' in device_name_lower and 'pipewire' not in device_name_lower):
                                    device_id = i
                                    print(f"\n✓ Using PulseAudio device [{i}] {device['name']} for monitor source")
                                    break
                    
                    # Prioritas 3: Coba match dengan "Built-in Audio" yang mungkin monitor
                    if device_id is None:
                        monitor_base = monitor_source_name.replace('.monitor', '').lower()
                        for i, device in enumerate(devices):
                            if device['max_input_channels'] > 0:
                                device_name_lower = device['name'].lower()
                                # Cari device yang mengandung "built-in" dan "audio" atau "analog stereo"
                                if (('built-in' in device_name_lower and 'audio' in device_name_lower) or
                                    ('analog' in device_name_lower and 'stereo' in device_name_lower)):
                                    device_id = i
                                    print(f"\n✓ Matched monitor device: [{i}] {device['name']}")
                                    break
            except Exception as e:
                print(f"Error deteksi PulseAudio/PipeWire: {str(e)}")
        
        # Fallback: cari device monitor di sounddevice (tanpa pulsectl)
        if device_id is None:
            print("\nTrying fallback: searching sounddevice for monitor devices...")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name_lower = device['name'].lower()
                    # Hanya ambil yang jelas-jelas monitor, hindari microphone
                    if (('monitor' in device_name_lower or 'loopback' in device_name_lower) and
                        'mic' not in device_name_lower and
                        'microphone' not in device_name_lower and
                        'input' not in device_name_lower):
                        device_id = i
                        print(f"Found monitor device: [{i}] {device['name']}")
                        break
        
        if device_id is None:
            print("\nWARNING: No monitor device found! System audio capture may not work.")
            print("Available input devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  [{i}] {device['name']}")
        
        return device_id

    def _find_windows_loopback_device(self):
        """Mencari device WASAPI loopback di Windows"""
        device_id = None
        
        print("\nSearching for Windows WASAPI loopback device...")
        
        if PYCAW_AVAILABLE:
            try:
                # Ambil default playback device
                devices = AudioUtilities.GetSpeakers()
                device_name = devices.FriendlyName if devices else None
                
                print(f"Default playback device: {device_name}")
                
                if device_name:
                    # Cari device yang cocok di sounddevice
                    sd_devices = sd.query_devices()
                    print(f"\nAvailable sounddevice input devices:")
                    for i, device in enumerate(sd_devices):
                        if device['max_input_channels'] > 0:
                            print(f"  [{i}] {device['name']} (channels: {device['max_input_channels']})")
                            device_name_lower = device['name'].lower()
                            # Device WASAPI loopback biasanya punya penamaan khusus
                            if ('loopback' in device_name_lower or 
                                device_name.lower() in device_name_lower or
                                ('wasapi' in device_name_lower and 'loopback' in device_name_lower)):
                                device_id = i
                                print(f"    -> MATCHED!")
                                break
            except Exception as e:
                print(f"Error deteksi WASAPI: {str(e)}")
        
        # Fallback: cari loopback di nama device
        if device_id is None:
            print("\nTrying fallback: searching for loopback devices...")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name_lower = device['name'].lower()
                    # Hindari microphone devices
                    if (('loopback' in device_name_lower or 
                         'wasapi' in device_name_lower or
                         'stereo mix' in device_name_lower) and
                        'mic' not in device_name_lower and
                        'microphone' not in device_name_lower):
                        device_id = i
                        print(f"Found loopback device: [{i}] {device['name']}")
                        break
        
        if device_id is None:
            print("\nWARNING: No WASAPI loopback device found!")
            print("You may need to enable 'Stereo Mix' in Windows Sound settings.")
        
        return device_id

    def start_capture(self, callback):
        """Memulai capture audio sistem - Cross-platform"""
        self.callback = callback
        device_id = None
        
        try:
            # Deteksi device khusus platform
            if IS_LINUX:
                device_id = self._find_linux_monitor_device()
            elif IS_WINDOWS:
                device_id = self._find_windows_loopback_device()
            
            # JANGAN fallback ke default input (biasanya microphone)
            # Jika tidak ada monitor device, gagal dengan jelas
            if device_id is None:
                print("\nERROR: No system audio monitor device found!")
                print("This is required for system audio capture (not microphone).")
                if IS_LINUX:
                    print("\nTroubleshooting for Linux:")
                    print("1. Make sure PipeWire or PulseAudio is running:")
                    print("   systemctl --user status pipewire")
                    print("   OR")
                    print("   pulseaudio --check")
                    print("2. Check available monitor sources:")
                    print("   pactl list sources short | grep monitor")
                    print("   OR")
                    print("   pw-cli list-objects | grep -i monitor")
                    print("3. You may need to create a virtual sink/monitor if none exists")
                elif IS_WINDOWS:
                    print("\nTroubleshooting for Windows:")
                    print("1. Enable 'Stereo Mix' in Windows Sound settings")
                    print("2. Right-click speaker icon -> Sounds -> Recording tab")
                    print("3. Enable 'Stereo Mix' if available")
                return False
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Status audio: {status}")
                if self.is_recording and self.callback:
                    # Konversi ke int16 (range [-32768, 32767])
                    # indata sudah dalam float32 range [-1.0, 1.0]
                    audio_int16 = (indata * 32768.0).clip(-32768, 32767).astype(np.int16)
                    self.callback(audio_int16.flatten())
            
            # Verifikasi device adalah input device
            device_info = sd.query_devices(device_id)
            if device_info['max_input_channels'] == 0:
                print(f"ERROR: Device [{device_id}] {device_info['name']} is not an input device!")
                return False
            
            self.audio_stream = sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            )
            
            self.is_recording = True
            self.audio_stream.start()
            
            print(f"\n✓ Successfully capturing from: {device_info['name']}")
            print(f"  Device ID: {device_id}, Channels: {self.channels}, Sample Rate: {self.sample_rate} Hz")
            
            return True
        except Exception as e:
            print(f"Error memulai capture audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def stop_capture(self):
        """Menghentikan capture audio"""
        self.is_recording = False
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass
        
        # Restore default source jika sebelumnya diubah
        if self.pulse and hasattr(self, '_old_default_source') and self._old_default_source:
            try:
                self.pulse.source_default_set(self._old_default_source)
                print(f"Restored default source to: {self._old_default_source}")
            except Exception as e:
                print(f"Warning: Could not restore default source: {str(e)}")
        
        if self.pulse:
            try:
                self.pulse.close()
            except:
                pass
        self.callback = None


class MainWindow(QMainWindow):
    """Jendela aplikasi utama"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Captioning Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Inisialisasi komponen
        self.caption_worker = None
        self.audio_capture = None
        self.worker_thread = None
        self.captions = []  # Simpan semua caption dengan timestamp
        self.is_capturing = False
        
        # Setup UI
        self.setup_ui()
        
        # Timer untuk proses real-time
        self.realtime_timer = QTimer()
        self.realtime_timer.timeout.connect(self.process_realtime_chunk)
        self.audio_buffer = []
        self.buffer_duration = 3.0  # Buffer 3 detik

    def setup_ui(self):
        """Menyiapkan antarmuka pengguna"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Judul
        title = QLabel("Auto Captioning Application")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Grup Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # Baris pertama: Model dan Language
        first_row = QHBoxLayout()
        
        # Pemilihan model
        first_row.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        first_row.addWidget(self.model_combo)
        
        first_row.addSpacing(20)  # Spacing antara model dan language
        
        # Pemilihan bahasa
        first_row.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        # Format: "Display Name (code)"
        self.language_combo.addItems([
            "English (en)",
            "Bahasa Indonesia (id)"
        ])
        self.language_combo.setCurrentText("English (en)")
        first_row.addWidget(self.language_combo)
        
        first_row.addStretch()
        settings_layout.addLayout(first_row)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Grup Pemrosesan File Video
        video_group = QGroupBox("Video File Processing")
        video_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_label)
        
        self.browse_btn = QPushButton("Browse Video File")
        self.browse_btn.clicked.connect(self.browse_video_file)
        file_layout.addWidget(self.browse_btn)
        
        self.process_btn = QPushButton("Process Video")
        self.process_btn.clicked.connect(self.process_video_file)
        self.process_btn.setEnabled(False)
        file_layout.addWidget(self.process_btn)
        
        video_layout.addLayout(file_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        video_layout.addWidget(self.progress_bar)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Grup Capture Audio Sistem
        system_group = QGroupBox("System Audio Capture (Real-time)")
        system_layout = QVBoxLayout()
        
        control_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Start System Audio Capture")
        self.capture_btn.clicked.connect(self.toggle_system_capture)
        control_layout.addWidget(self.capture_btn)
        
        self.clear_btn = QPushButton("Clear Captions")
        self.clear_btn.clicked.connect(self.clear_captions)
        control_layout.addWidget(self.clear_btn)
        
        system_layout.addLayout(control_layout)
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        # Tampilan Caption
        caption_group = QGroupBox("Captions")
        caption_layout = QVBoxLayout()
        
        self.caption_display = QTextEdit()
        self.caption_display.setReadOnly(True)
        self.caption_display.setFont(QFont("Arial", 12))
        caption_layout.addWidget(self.caption_display)
        
        # Tombol export
        export_layout = QHBoxLayout()
        self.export_txt_btn = QPushButton("Export as Text File")
        self.export_txt_btn.clicked.connect(self.export_text_file)
        export_layout.addWidget(self.export_txt_btn)
        
        self.export_srt_btn = QPushButton("Export as SRT (Subtitle)")
        self.export_srt_btn.clicked.connect(self.export_srt_file)
        export_layout.addWidget(self.export_srt_btn)
        
        caption_layout.addLayout(export_layout)
        caption_group.setLayout(caption_layout)
        layout.addWidget(caption_group)
        
        layout.addStretch()

    def browse_video_file(self):
        """Mencari file video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)

    def process_video_file(self):
        """Memproses file video dan mengekstrak caption"""
        if not hasattr(self, 'video_path'):
            QMessageBox.warning(self, "Error", "Please select a video file first")
            return
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Ekstrak audio dari video
        audio_path = None
        try:
            video = VideoFileClip(self.video_path)
            
            if video.audio is None:
                raise Exception("Video has no audio track")
            
            # Buat path untuk file audio temporary
            audio_path = os.path.join(
                os.path.dirname(self.video_path) if os.path.dirname(self.video_path) else os.getcwd(),
                f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            print(f"Extracting audio to: {audio_path}")
            
            # Extract audio dengan format yang kompatibel dengan Whisper
            # MoviePy akan extract sebagai WAV, Whisper akan handle resampling otomatis
            video.audio.write_audiofile(audio_path)
            
            video.close()
            
            # Validasi file audio yang diekstrak
            if not os.path.exists(audio_path):
                raise Exception("Audio file was not created")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise Exception("Extracted audio file is empty")
            
            print(f"Audio extracted successfully: {file_size} bytes")
            
        except Exception as e:
            error_msg = f"Failed to extract audio: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", error_msg)
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            # Cleanup jika file dibuat tapi error
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            return
        
        # Proses audio
        language_code = self.get_language_code()
        self.caption_worker = CaptionWorker(
            self.model_combo.currentText(),
            language=language_code
        )
        self.worker_thread = threading.Thread(
            target=self.caption_worker.process_audio_file,
            args=(audio_path,)
        )
        
        self.caption_worker.caption_ready.connect(self.add_caption)
        self.caption_worker.progress_update.connect(self.progress_bar.setValue)
        self.caption_worker.finished.connect(self.on_processing_finished)
        self.caption_worker.error.connect(self.on_processing_error)
        
        self.worker_thread.start()
        
        # Bersihkan file audio sementara setelah proses
        def cleanup():
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        self.caption_worker.finished.connect(cleanup)

    def get_language_code(self):
        """Mengambil kode bahasa dari combobox"""
        language_text = self.language_combo.currentText()
        # Format: "English (en)" -> extract "en"
        if "(" in language_text and ")" in language_text:
            return language_text.split("(")[1].split(")")[0]
        return "en"  # Default ke English
    
    def toggle_system_capture(self):
        """Toggle capture audio sistem"""
        if not self.is_capturing:
            self.start_system_capture()
        else:
            self.stop_system_capture()

    def start_system_capture(self):
        """Memulai capture audio sistem"""
        self.is_capturing = True
        self.capture_btn.setText("Stop System Audio Capture")
        self.capture_btn.setStyleSheet("background-color: #ff4444;")
        
        # Inisialisasi worker dengan language yang dipilih
        language_code = self.get_language_code()
        self.caption_worker = CaptionWorker(
            self.model_combo.currentText(),
            language=language_code
        )
        self.caption_worker.caption_ready.connect(self.add_caption)
        self.caption_worker.error.connect(self.on_processing_error)
        
        # Muat model di background
        threading.Thread(target=self.caption_worker.load_model, daemon=True).start()
        
        # Inisialisasi capture audio
        self.audio_capture = AudioCapture()
        
        def audio_callback(audio_data):
            if self.is_capturing:
                self.audio_buffer.append(audio_data)
                # Simpan hanya 3 detik terakhir
                max_samples = int(16000 * self.buffer_duration)
                total_samples = sum(len(chunk) for chunk in self.audio_buffer)
                while total_samples > max_samples:
                    removed = self.audio_buffer.pop(0)
                    total_samples -= len(removed)
        
        if self.audio_capture.start_capture(audio_callback):
            # Mulai timer proses
            self.realtime_timer.start(3000)  # Proses setiap 3 detik
        else:
            platform_msg = ""
            if IS_LINUX:
                platform_msg = "Make sure PipeWire or PulseAudio is running."
            elif IS_WINDOWS:
                platform_msg = "Make sure WASAPI is available. You may need to enable 'Stereo Mix' in Windows sound settings."
            else:
                platform_msg = "Check your audio system configuration."
            
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start audio capture. {platform_msg}"
            )
            self.stop_system_capture()

    def stop_system_capture(self):
        """Menghentikan capture audio sistem"""
        self.is_capturing = False
        self.capture_btn.setText("Start System Audio Capture")
        self.capture_btn.setStyleSheet("")
        
        if self.realtime_timer.isActive():
            self.realtime_timer.stop()
        
        if self.audio_capture:
            self.audio_capture.stop_capture()
        
        if self.caption_worker:
            self.caption_worker.stop()

    def process_realtime_chunk(self):
        """Memproses buffer audio yang terkumpul"""
        if not self.audio_buffer:
            return
        
        if not self.caption_worker:
            return
        
        # Pastikan model sudah dimuat
        if not self.caption_worker.model:
            print("Model not loaded yet, waiting...")
            return
        
        # Gabungkan buffer audio
        if self.audio_buffer:
            try:
                audio_chunk = np.concatenate(self.audio_buffer)
                self.audio_buffer.clear()
                
                # Pastikan ada data audio
                if len(audio_chunk) > 0:
                    # Proses di thread background
                    threading.Thread(
                        target=self.caption_worker.process_realtime_audio,
                        args=(audio_chunk,),
                        daemon=True
                    ).start()
                else:
                    print("Warning: Empty audio chunk, skipping...")
            except Exception as e:
                print(f"Error concatenating audio buffer: {str(e)}")
                self.audio_buffer.clear()

    def add_caption(self, text, start_time, end_time):
        """Menambahkan caption ke tampilan"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        time_str = f"[{start_time:.1f}s - {end_time:.1f}s]"
        
        # Simpan caption
        self.captions.append({
            "text": text,
            "start": start_time,
            "end": end_time,
            "timestamp": timestamp
        })
        
        # Perbarui tampilan
        self.caption_display.append(f"{timestamp} {time_str}: {text}")
        
        # Auto-scroll ke bawah
        cursor = self.caption_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.caption_display.setTextCursor(cursor)

    def clear_captions(self):
        """Menghapus semua caption"""
        self.caption_display.clear()
        self.captions = []

    def export_text_file(self):
        """Ekspor caption sebagai file teks biasa"""
        if not self.captions:
            QMessageBox.warning(self, "Error", "No captions to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Text File",
            f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for caption in self.captions:
                        f.write(f"[{caption['start']:.2f}s - {caption['end']:.2f}s] {caption['text']}\n")
                QMessageBox.information(self, "Success", f"Captions exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def export_srt_file(self):
        """Ekspor caption sebagai file subtitle SRT"""
        if not self.captions:
            QMessageBox.warning(self, "Error", "No captions to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SRT File",
            f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
            "SRT Files (*.srt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for i, caption in enumerate(self.captions, 1):
                        start_time = self.format_srt_time(caption['start'])
                        end_time = self.format_srt_time(caption['end'])
                        f.write(f"{i}\n")
                        f.write(f"{start_time} --> {end_time}\n")
                        f.write(f"{caption['text']}\n\n")
                QMessageBox.information(self, "Success", f"SRT file exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def format_srt_time(self, seconds):
        """Format detik ke format waktu SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def on_processing_finished(self):
        """Dipanggil ketika pemrosesan video selesai"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "Success", "Video processing completed!")

    def on_processing_error(self, error_msg):
        """Menangani error pemrosesan"""
        QMessageBox.critical(self, "Error", error_msg)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.is_capturing:
            self.stop_system_capture()

    def closeEvent(self, event):
        """Menangani event penutupan jendela"""
        if self.is_capturing:
            self.stop_system_capture()
        if self.caption_worker:
            self.caption_worker.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
