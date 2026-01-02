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
    QGroupBox, QComboBox, QSpinBox, QCheckBox, QProgressBar,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSlider, QFormLayout, QColorDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint, QPointF, QEvent
from PyQt6.QtGui import QFont, QTextCursor, QColor, QIcon, QPainter, QPen

import whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    FasterWhisperModel = None

import sounddevice as sd
import numpy as np
from moviepy import VideoFileClip
import requests
import hashlib

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

# Deteksi Wayland vs X11 (untuk Linux)
IS_WAYLAND = False
if IS_LINUX:
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    xdg_session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    IS_WAYLAND = bool(wayland_display) or xdg_session == "wayland"


class CaptionWorker(QObject):
    """Kelas worker untuk memproses audio dan menghasilkan caption"""
    caption_ready = pyqtSignal(str, float, float)  # teks, waktu_mulai, waktu_selesai
    progress_update = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_name="base", language="en", engine="openai"):
        super().__init__()
        self.model = None
        self.model_name = model_name
        self.language = language
        self.engine = engine  # "openai" or "faster"
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # Proses chunk 3 detik

    def load_model(self):
        """Memuat model Whisper berdasarkan engine yang dipilih"""
        try:
            if self.engine == "faster":
                if not FASTER_WHISPER_AVAILABLE:
                    raise ImportError("faster-whisper is not installed. Please install it with: pip install faster-whisper")
                
                # Faster Whisper menggunakan device="cpu" atau "cuda" otomatis
                # compute_type bisa "int8", "int8_float16", "float16", "float32"
                import torch
                
                # Coba CUDA dulu, tapi dengan error handling untuk cuDNN issues
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                
                print(f"Loading Faster Whisper model: {self.model_name} (device: {device}, compute_type: {compute_type})")
                
                # Cek apakah model sudah didownload
                from pathlib import Path
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub") if not IS_WINDOWS else os.path.join(os.environ.get("LOCALAPPDATA", ""), "huggingface", "hub")
                model_dir = os.path.join(cache_dir, f"models--guillaumekln--faster-whisper-{self.model_name}")
                if os.path.exists(model_dir):
                    print(f"Model directory found: {model_dir}")
                else:
                    print(f"Model directory not found, will download automatically: {model_dir}")
                
                # Coba load dengan CUDA, tapi test dulu apakah cuDNN tersedia
                # Jika cuDNN tidak tersedia, langsung gunakan CPU
                if device == "cuda":
                    # Test apakah cuDNN tersedia dengan mencoba import
                    try:
                        import torch
                        # Cek apakah cuDNN tersedia
                        if not torch.backends.cudnn.is_available():
                            print("Warning: cuDNN is not available, using CPU instead")
                            device = "cpu"
                            compute_type = "int8"
                            self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type)
                            print(f"✓ Faster Whisper model '{self.model_name}' loaded successfully on CPU")
                        else:
                            # Coba load dengan CUDA
                            self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type)
                            print(f"✓ Faster Whisper model '{self.model_name}' loaded successfully on {device}")
                            # Test inference sederhana untuk detect cuDNN error lebih awal
                            try:
                                import numpy as np
                                test_audio = np.zeros((16000,), dtype=np.float32)  # 1 second of silence
                                list(self.model.transcribe(test_audio, beam_size=1))
                                print("✓ Model test inference successful")
                            except Exception as test_error:
                                error_str = str(test_error).lower()
                                if "cudnn" in error_str or "libcudnn" in error_str or "invalid handle" in error_str:
                                    print(f"Warning: cuDNN error detected during test: {test_error}")
                                    print("Reloading with CPU...")
                                    self.model = None
                                    device = "cpu"
                                    compute_type = "int8"
                                    self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type)
                                    print(f"✓ Faster Whisper model '{self.model_name}' reloaded on CPU")
                    except Exception as cuda_error:
                        error_str = str(cuda_error).lower()
                        # Jika error terkait CUDA/cuDNN, fallback ke CPU
                        if "cudnn" in error_str or "cuda" in error_str or "invalid handle" in error_str or "libcudnn" in error_str:
                            print(f"Warning: CUDA/cuDNN error detected: {cuda_error}")
                            print("Falling back to CPU...")
                            device = "cpu"
                            compute_type = "int8"
                            try:
                                self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type)
                                print(f"✓ Faster Whisper model '{self.model_name}' loaded successfully on CPU (fallback)")
                            except Exception as cpu_error:
                                print(f"Error loading on CPU: {cpu_error}")
                                raise
                        else:
                            # Re-raise error jika bukan CUDA/cuDNN issue
                            raise
                else:
                    # Langsung load dengan CPU
                    self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type)
                    print(f"✓ Faster Whisper model '{self.model_name}' loaded successfully on {device}")
            else:  # openai
                self.model = whisper.load_model(self.model_name)
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

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
            
            # Transcribe berdasarkan engine
            if self.engine == "faster":
                # Faster Whisper bisa langsung dari file path
                segments, info = self.model.transcribe(
                    audio_path,
                    language=self.language if self.language != "en" else None,
                    beam_size=5
                )
                
                # Convert segments generator ke list untuk progress tracking
                segments_list = list(segments)
                
                if not segments_list:
                    self.error.emit("No speech detected in audio file")
                    return
                
                # Emit segments
                total_duration = segments_list[-1].end if segments_list else 1.0
                for segment in segments_list:
                    if not self.is_running:
                        break
                    text = segment.text.strip()
                    if text:
                        self.caption_ready.emit(
                            text,
                            segment.start,
                            segment.end
                        )
                        # Update progress
                        progress = int((segment.end / total_duration) * 100)
                        self.progress_update.emit(progress)
            else:  # openai
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
            
            # Transcribe berdasarkan engine
            if self.engine == "faster":
                # Faster Whisper memerlukan audio dalam format numpy array
                # Audio sudah dalam format float32 [-1.0, 1.0]
                try:
                    segments, info = self.model.transcribe(
                        audio_float,
                        language=self.language if self.language != "en" else None,
                        beam_size=5
                    )
                    
                    # Convert segments generator ke list
                    segments_list = list(segments)
                    
                    # Debug: print jumlah segments
                    if segments_list:
                        print(f"Faster Whisper: Found {len(segments_list)} segments")
                    
                    for segment in segments_list:
                        text = segment.text.strip()
                        # Filter out invalid segments:
                        # - Empty text
                        # - Single word "You" (common false positive)
                        # - Very short segments (< 2 characters)
                        # - Only punctuation
                        if text and len(text) >= 2:
                            # Skip common false positives
                            text_lower = text.lower()
                            if text_lower in ["you", "uh", "um", "ah", "eh", "oh", "hmm", "mm"]:
                                continue
                            # Skip if only punctuation
                            if text.strip(".,!?;: ") == "":
                                continue
                            print(f"Faster Whisper: Emitting caption: '{text}' (start={segment.start:.2f}, end={segment.end:.2f})")
                            self.caption_ready.emit(text, segment.start, segment.end)
                except Exception as transcribe_error:
                    error_str = str(transcribe_error).lower()
                    # Jika error terkait CUDA/cuDNN saat transcribe, reload dengan CPU
                    if "cudnn" in error_str or "libcudnn" in error_str or "invalid handle" in error_str or "cuda" in error_str:
                        print(f"Warning: CUDA/cuDNN error during transcription: {transcribe_error}")
                        print("Reloading model with CPU...")
                        # Reload model dengan CPU
                        try:
                            self.model = None
                            import torch
                            device = "cpu"
                            compute_type = "int8"
                            self.model = FasterWhisperModel(self.model_name, device=device, compute_type=compute_type)
                            print(f"✓ Faster Whisper model '{self.model_name}' reloaded on CPU")
                            # Retry transcribe dengan CPU model
                            segments, info = self.model.transcribe(
                                audio_float,
                                language=self.language if self.language != "en" else None,
                                beam_size=5
                            )
                            segments_list = list(segments)
                            for segment in segments_list:
                                text = segment.text.strip()
                                # Filter out invalid segments (same as above)
                                if text and len(text) >= 2:
                                    text_lower = text.lower()
                                    if text_lower in ["you", "uh", "um", "ah", "eh", "oh", "hmm", "mm"]:
                                        continue
                                    if text.strip(".,!?;: ") == "":
                                        continue
                                    self.caption_ready.emit(text, segment.start, segment.end)
                        except Exception as cpu_error:
                            print(f"Error reloading on CPU: {cpu_error}")
                            self.error.emit(f"Error processing audio: {cpu_error}")
                    else:
                        # Re-raise jika bukan CUDA/cuDNN error
                        raise
            else:  # openai
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


class ModelManagerDialog(QDialog):
    """Dialog untuk mengelola model Whisper"""
    download_success = pyqtSignal(str, bool, str)  # model_name, success, error_msg
    download_progress_update = pyqtSignal(int)  # progress value (0-100)
    download_size_update = pyqtSignal(int, int)  # downloaded_bytes, total_bytes
    
    def __init__(self, parent=None, engine="openai"):
        super().__init__(parent)
        self.parent_window = parent  # Simpan referensi ke parent untuk akses translation
        self.engine = engine  # "openai" or "faster"
        self.setGeometry(200, 200, 700, 500)
        
        # Update window title berdasarkan engine
        if self.engine == "faster":
            self.setWindowTitle("Faster Whisper Model Manager")
        else:
            self.setWindowTitle("OpenAI Whisper Model Manager")
        
        # Model yang tersedia dengan translation
        self.available_models_en = {
            "tiny": "~39M parameters - Fastest, least accurate",
            "base": "~74M parameters - Good balance (Recommended)",
            "small": "~244M parameters - Better accuracy",
            "medium": "~769M parameters - High accuracy",
            "large": "~1550M parameters - Best accuracy, slowest"
        }
        self.available_models_id = {
            "tiny": "~39M parameter - Tercepat, akurasi terendah",
            "base": "~74M parameter - Keseimbangan baik (Direkomendasikan)",
            "small": "~244M parameter - Akurasi lebih baik",
            "medium": "~769M parameter - Akurasi tinggi",
            "large": "~1550M parameter - Akurasi terbaik, paling lambat"
        }
        
        # Translation dictionary untuk ModelManagerDialog
        self.translations = {
            "en": {
                "window_title": "Whisper Model Manager",
                "downloaded_models": "Downloaded Models:",
                "model_name": "Model Name",
                "status": "Status",
                "size": "Size",
                "actions": "Actions",
                "downloaded": "Downloaded",
                "not_downloaded": "Not Downloaded",
                "downloading": "Downloading...",
                "delete": "Delete",
                "download": "Download",
                "download_new_model": "Download New Model",
                "select_model": "Select Model:",
                "cancel": "Cancel",
                "cancelling": "Cancelling...",
                "close": "Close",
                "download_in_progress": "Download in Progress",
                "model_downloading": "Model '{model_name}' is currently being downloaded. Please wait for it to complete.",
                "model_already_exists": "Model Already Exists",
                "model_already_downloaded": "Model '{model_name}' is already downloaded. Do you want to re-download it?",
                "confirm_delete": "Confirm Delete",
                "delete_confirmation": "Are you sure you want to delete model '{model_name}'?",
                "delete_warning": "This action cannot be undone.",
                "success": "Success",
                "model_downloaded": "Model '{model_name}' downloaded successfully!",
                "model_deleted": "Model '{model_name}' deleted successfully!",
                "error": "Error",
                "download_failed": "Failed to download model '{model_name}':\n{error_msg}",
                "delete_failed": "Failed to delete model: {error_msg}",
                "warning": "Warning",
                "model_not_found": "Model file not found: {model_path}",
                "cancel_download": "Cancel Download",
                "cancel_confirmation": "Are you sure you want to cancel the download?"
            },
            "id": {
                "window_title": "Pengelola Model Whisper",
                "downloaded_models": "Model yang Diunduh:",
                "model_name": "Nama Model",
                "status": "Status",
                "size": "Ukuran",
                "actions": "Aksi",
                "downloaded": "Diunduh",
                "not_downloaded": "Belum Diunduh",
                "downloading": "Mengunduh...",
                "delete": "Hapus",
                "download": "Unduh",
                "download_new_model": "Unduh Model Baru",
                "select_model": "Pilih Model:",
                "cancel": "Batal",
                "cancelling": "Membatalkan...",
                "close": "Tutup",
                "download_in_progress": "Unduhan Sedang Berlangsung",
                "model_downloading": "Model '{model_name}' sedang diunduh. Silakan tunggu hingga selesai.",
                "model_already_exists": "Model Sudah Ada",
                "model_already_downloaded": "Model '{model_name}' sudah diunduh. Apakah Anda ingin mengunduh ulang?",
                "confirm_delete": "Konfirmasi Hapus",
                "delete_confirmation": "Apakah Anda yakin ingin menghapus model '{model_name}'?",
                "delete_warning": "Tindakan ini tidak dapat dibatalkan.",
                "success": "Berhasil",
                "model_downloaded": "Model '{model_name}' berhasil diunduh!",
                "model_deleted": "Model '{model_name}' berhasil dihapus!",
                "error": "Kesalahan",
                "download_failed": "Gagal mengunduh model '{model_name}':\n{error_msg}",
                "delete_failed": "Gagal menghapus model: {error_msg}",
                "warning": "Peringatan",
                "model_not_found": "File model tidak ditemukan: {model_path}",
                "cancel_download": "Batalkan Unduhan",
                "cancel_confirmation": "Apakah Anda yakin ingin membatalkan unduhan?"
            }
        }
        
        # Flag untuk cancel download dan track downloading models
        # HARUS diinisialisasi SEBELUM setup_ui() dan refresh_model_list()
        self.download_cancelled = False
        self.download_thread = None
        self.downloading_models = {}  # Track model yang sedang didownload {model_name: True}
        self.download_session = None  # Requests session untuk bisa dihentikan
        self.download_response = None  # Response object untuk bisa dihentikan
        
        # Setup UI
        self.setup_ui()
        self.refresh_model_list()
        # Update UI language setelah setup (akan dipanggil setelah setup_ui)
    
    def get_ui_language(self):
        """Mendapatkan bahasa UI dari parent atau default ke English"""
        if self.parent_window and hasattr(self.parent_window, 'ui_language'):
            return self.parent_window.ui_language
        return "en"
    
    def tr(self, key, **kwargs):
        """Mendapatkan terjemahan untuk key tertentu"""
        lang = self.get_ui_language()
        translation = self.translations.get(lang, self.translations["en"]).get(key, key)
        # Format string jika ada kwargs
        if kwargs:
            try:
                return translation.format(**kwargs)
            except:
                return translation
        return translation
    
    def update_ui_language(self):
        """Update semua teks UI berdasarkan bahasa yang dipilih"""
        # Window title
        self.setWindowTitle(self.tr("window_title"))
        
        # Title label
        if hasattr(self, 'title_label'):
            self.title_label.setText(self.tr("window_title"))
        
        # Table label
        if hasattr(self, 'table_label'):
            self.table_label.setText(self.tr("downloaded_models"))
        
        # Table headers
        if hasattr(self, 'model_table'):
            self.model_table.setHorizontalHeaderLabels([
                self.tr("model_name"),
                self.tr("status"),
                self.tr("size"),
                self.tr("actions")
            ])
        
        # Download group
        if hasattr(self, 'download_group'):
            self.download_group.setTitle(self.tr("download_new_model"))
        
        # Download label
        if hasattr(self, 'select_model_label'):
            self.select_model_label.setText(self.tr("select_model"))
        
        # Buttons
        if hasattr(self, 'download_btn'):
            self.download_btn.setText(self.tr("download"))
        if hasattr(self, 'cancel_btn'):
            self.cancel_btn.setText(self.tr("cancel"))
        if hasattr(self, 'close_btn'):
            self.close_btn.setText(self.tr("close"))
        
        # Update model info
        if hasattr(self, 'download_combo'):
            self.update_model_info()
        
        # Refresh table untuk update status text
        self.refresh_model_list()
    
    def setup_ui(self):
        """Setup UI untuk model manager"""
        layout = QVBoxLayout(self)
        
        # Judul
        self.title_label = QLabel(self.tr("window_title"))
        self.title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Tabel model
        self.table_label = QLabel(self.tr("downloaded_models"))
        layout.addWidget(self.table_label)
        
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels([
            self.tr("model_name"),
            self.tr("status"),
            self.tr("size"),
            self.tr("actions")
        ])
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.model_table)
        
        # Download section
        self.download_group = QGroupBox(self.tr("download_new_model"))
        download_layout = QVBoxLayout()
        
        download_row = QHBoxLayout()
        self.select_model_label = QLabel(self.tr("select_model"))
        download_row.addWidget(self.select_model_label)
        self.download_combo = QComboBox()
        self.download_combo.addItems(list(self.available_models_en.keys()))
        download_row.addWidget(self.download_combo)
        
        self.download_btn = QPushButton(self.tr("download"))
        self.download_btn.clicked.connect(self.download_model)
        download_row.addWidget(self.download_btn)
        download_layout.addLayout(download_row)
        
        # Info model
        self.model_info_label = QLabel()
        self.model_info_label.setWordWrap(True)
        download_layout.addWidget(self.model_info_label)
        
        # Progress bar untuk download
        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)
        download_layout.addWidget(self.download_progress)
        
        # Label untuk menampilkan progress ukuran (X MB / Y MB)
        self.download_size_label = QLabel()
        self.download_size_label.setVisible(False)
        self.download_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        download_layout.addWidget(self.download_size_label)
        
        # Tombol cancel
        cancel_row = QHBoxLayout()
        cancel_row.addStretch()
        self.cancel_btn = QPushButton(self.tr("cancel"))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setStyleSheet("background-color: #ff4444; color: white;")
        self.cancel_btn.clicked.connect(self.cancel_download)
        cancel_row.addWidget(self.cancel_btn)
        download_layout.addLayout(cancel_row)
        
        self.download_group.setLayout(download_layout)
        layout.addWidget(self.download_group)
        
        # Update info saat model dipilih
        self.download_combo.currentTextChanged.connect(self.update_model_info)
        
        # Connect signals untuk download (thread-safe UI updates)
        self.download_success.connect(self.on_download_complete)
        self.download_progress_update.connect(self.update_download_progress)
        self.download_size_update.connect(self.update_download_size)
        self.download_size_update.connect(self.update_download_size)
        
        # Tombol close
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.close_btn = QPushButton(self.tr("close"))
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
    
    def update_model_info(self, model_name=None):
        """Update info model saat dipilih"""
        if model_name is None:
            model_name = self.download_combo.currentText()
        
        lang = self.get_ui_language()
        available_models = self.available_models_id if lang == "id" else self.available_models_en
        
        if model_name in available_models:
            self.model_info_label.setText(available_models[model_name])
    
    def get_model_cache_path(self):
        """Mendapatkan path cache untuk model berdasarkan engine"""
        if self.engine == "faster":
            # Faster Whisper menggunakan Hugging Face cache
            if IS_WINDOWS:
                cache_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "huggingface", "hub")
            else:
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            return cache_dir
        else:  # openai
            if IS_WINDOWS:
                cache_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "whisper")
            else:
                cache_dir = os.path.expanduser("~/.cache/whisper")
            return cache_dir
    
    def get_model_file_path(self, model_name):
        """Mendapatkan path file model berdasarkan engine"""
        if self.engine == "faster":
            # Faster Whisper models di-download otomatis saat pertama kali digunakan
            # Model disimpan di Hugging Face cache dengan format: models--guillaumekln--faster-whisper-{model_name}
            cache_dir = self.get_model_cache_path()
            model_dir = os.path.join(cache_dir, f"models--guillaumekln--faster-whisper-{model_name}")
            # Check if model directory exists
            if os.path.exists(model_dir):
                # Check snapshots directory
                snapshots_dir = os.path.join(model_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Find the first snapshot directory (usually contains the model files)
                    for snapshot_name in os.listdir(snapshots_dir):
                        snapshot_path = os.path.join(snapshots_dir, snapshot_name)
                        if os.path.isdir(snapshot_path):
                            # Check if this snapshot contains model files
                            for root, dirs, files in os.walk(snapshot_path):
                                if "model.bin" in files or "model.safetensors" in files or "config.json" in files:
                                    return snapshot_path
                # Fallback: check entire model_dir for model files
                for root, dirs, files in os.walk(model_dir):
                    if "model.bin" in files or "model.safetensors" in files or "config.json" in files:
                        return root
            return model_dir  # Return directory path even if not exists yet
        else:  # openai
            cache_dir = self.get_model_cache_path()
            return os.path.join(cache_dir, f"{model_name}.pt")
    
    def format_file_size(self, size_bytes):
        """Format ukuran file menjadi readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def refresh_model_list(self):
        """Refresh daftar model yang sudah terdownload"""
        self.model_table.setRowCount(0)
        cache_dir = self.get_model_cache_path()
        
        if not os.path.exists(cache_dir):
            return
        
        # Cek setiap model yang tersedia
        for model_name in self.available_models_en.keys():
            model_path = self.get_model_file_path(model_name)
            row = self.model_table.rowCount()
            self.model_table.insertRow(row)
            
            # Model name
            name_item = QTableWidgetItem(model_name)
            self.model_table.setItem(row, 0, name_item)
            
            # Status
            if model_name in self.downloading_models and self.downloading_models[model_name]:
                # Sedang didownload
                status_item = QTableWidgetItem("Downloading...")
                status_item.setForeground(Qt.GlobalColor.blue)
                size_item = QTableWidgetItem("In Progress")
            elif os.path.exists(model_path):
                status_item = QTableWidgetItem("Downloaded")
                status_item.setForeground(Qt.GlobalColor.green)
                # Calculate total size (for Faster Whisper, it's a directory)
                if os.path.isdir(model_path):
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(model_path)
                        for filename in filenames
                    )
                else:
                    total_size = os.path.getsize(model_path)
                size_item = QTableWidgetItem(self.format_file_size(total_size))
            else:
                status_item = QTableWidgetItem("Not Downloaded")
                status_item.setForeground(Qt.GlobalColor.red)
                size_item = QTableWidgetItem("N/A")
            
            self.model_table.setItem(row, 1, status_item)
            self.model_table.setItem(row, 2, size_item)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            if model_name in self.downloading_models and self.downloading_models[model_name]:
                # Sedang didownload - disable button
                downloading_label = QLabel("Downloading...")
                downloading_label.setStyleSheet("color: blue; font-weight: bold;")
                actions_layout.addWidget(downloading_label)
            elif os.path.exists(model_path):
                delete_btn = QPushButton("Delete")
                delete_btn.clicked.connect(lambda checked, name=model_name: self.delete_model(name))
                delete_btn.setStyleSheet("background-color: #ff4444; color: white;")
                actions_layout.addWidget(delete_btn)
            else:
                download_btn = QPushButton(self.tr("download"))
                download_btn.clicked.connect(lambda checked, name=model_name: self.download_model(name))
                actions_layout.addWidget(download_btn)
            
            actions_layout.addStretch()
            self.model_table.setCellWidget(row, 3, actions_widget)
    
    def download_model(self, model_name=None):
        """Download model Whisper"""
        if model_name is None:
            model_name = self.download_combo.currentText()
        
        # Cek apakah sedang didownload
        if model_name in self.downloading_models and self.downloading_models[model_name]:
            QMessageBox.information(
                self,
                self.tr("download_in_progress"),
                self.tr("model_downloading", model_name=model_name)
            )
            return
        
        model_path = self.get_model_file_path(model_name)
        
        # Cek apakah sudah terdownload
        if os.path.exists(model_path):
            reply = QMessageBox.question(
                self,
                self.tr("model_already_exists"),
                self.tr("model_already_downloaded", model_name=model_name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Mark model sebagai sedang didownload
        self.downloading_models[model_name] = True
        self.refresh_model_list()  # Refresh untuk update status
        
        # Reset cancel flag
        self.download_cancelled = False
        
        # Handle Faster Whisper download dengan progress tracking
        if self.engine == "faster":
            # Faster Whisper menggunakan Hugging Face Hub untuk download
            # Kita perlu download dengan progress tracking
            self.download_btn.setEnabled(False)
            self.download_progress.setVisible(True)
            self.download_progress.setValue(0)
            self.download_size_label.setVisible(True)
            self.download_size_label.setText("Preparing download...")
            self.cancel_btn.setVisible(True)
            
            # Mark model sebagai sedang didownload
            self.downloading_models[model_name] = True
            self.refresh_model_list()
            
            def download_faster_whisper_thread():
                try:
                    print(f"Downloading Faster Whisper model: {model_name}")
                    
                    # Download model dari Hugging Face menggunakan requests (seperti OpenAI Whisper)
                    repo_id = f"guillaumekln/faster-whisper-{model_name}"
                    cache_dir = self.get_model_cache_path()
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    # Struktur direktori Hugging Face cache
                    model_dir_name = f"models--guillaumekln--faster-whisper-{model_name}"
                    model_dir_path = os.path.join(cache_dir, model_dir_name)
                    
                    # Hapus direktori lama jika ada (untuk re-download)
                    if os.path.exists(model_dir_path):
                        import shutil
                        try:
                            shutil.rmtree(model_dir_path)
                        except:
                            pass
                    
                    # Dapatkan list file dari Hugging Face API
                    api_url = f"https://huggingface.co/api/models/{repo_id}"
                    session = requests.Session()
                    self.download_session = session
                    
                    # Check cancel sebelum request
                    if self.download_cancelled:
                        session.close()
                        self.download_session = None
                        raise Exception("Download cancelled by user")
                    
                    # Get model info dari API
                    print(f"Fetching model info from: {api_url}")
                    response = session.get(api_url, timeout=30)
                    response.raise_for_status()
                    model_info = response.json()
                    
                    # Dapatkan list file yang perlu didownload
                    siblings = model_info.get("siblings", [])
                    if not siblings:
                        raise Exception("No files found in model repository")
                    
                    # Filter file yang perlu didownload (skip .gitattributes, README, dll)
                    files_to_download = [
                        f for f in siblings 
                        if f.get("rfilename") and 
                        not f.get("rfilename").startswith(".") and
                        f.get("rfilename") != "README.md"
                    ]
                    
                    if not files_to_download:
                        raise Exception("No model files found to download")
                    
                    # Hitung total size
                    total_size = sum(f.get("size", 0) for f in files_to_download)
                    if total_size == 0:
                        # Estimasi jika size tidak tersedia
                        estimated_sizes = {
                            "tiny": 75 * 1024 * 1024,
                            "base": 150 * 1024 * 1024,
                            "small": 500 * 1024 * 1024,
                            "medium": 1500 * 1024 * 1024,
                            "large": 3000 * 1024 * 1024
                        }
                        total_size = estimated_sizes.get(model_name, 500 * 1024 * 1024)
                    
                    self.download_size_update.emit(0, total_size)
                    self.download_progress_update.emit(0)
                    
                    # Download setiap file
                    downloaded_total = 0
                    chunk_size = 8192  # 8KB chunks
                    
                    # Buat struktur direktori Hugging Face cache
                    snapshots_dir = os.path.join(model_dir_path, "snapshots")
                    os.makedirs(snapshots_dir, exist_ok=True)
                    
                    # Dapatkan commit hash dari API (atau gunakan "main" sebagai default)
                    # Hugging Face menggunakan commit hash sebagai snapshot ID
                    commit_hash = model_info.get("sha", "main")
                    # Jika sha tidak ada, coba dari siblings
                    if commit_hash == "main" and siblings:
                        # Coba dapatkan dari file pertama
                        first_file = siblings[0]
                        commit_hash = first_file.get("blob_id", "main")
                    
                    # Gunakan commit hash sebagai snapshot ID (atau "main" jika tidak ada)
                    snapshot_id = commit_hash if commit_hash and commit_hash != "main" else "main"
                    snapshot_path = os.path.join(snapshots_dir, snapshot_id)
                    os.makedirs(snapshot_path, exist_ok=True)
                    
                    for file_info in files_to_download:
                        # Check cancel sebelum setiap file
                        if self.download_cancelled:
                            print(f"Cancelling download for {model_name}...")
                            session.close()
                            self.download_session = None
                            # Hapus direktori yang tidak lengkap
                            if os.path.exists(model_dir_path):
                                import shutil
                                try:
                                    shutil.rmtree(model_dir_path)
                                    print(f"Removed incomplete directory: {model_dir_path}")
                                except:
                                    pass
                            if model_name in self.downloading_models:
                                del self.downloading_models[model_name]
                            self.download_success.emit(model_name, False, "Download cancelled by user")
                            return
                        
                        filename = file_info.get("rfilename")
                        file_size = file_info.get("size", 0)
                        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                        
                        file_path = os.path.join(snapshot_path, filename)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        print(f"Downloading {filename} ({self.format_file_size(file_size)})...")
                        
                        # Download file dengan streaming
                        file_response = session.get(file_url, stream=True, timeout=60)
                        file_response.raise_for_status()
                        self.download_response = file_response
                        
                        file_downloaded = 0
                        with open(file_path, 'wb') as f:
                            for chunk in file_response.iter_content(chunk_size=chunk_size):
                                # Check cancel setiap chunk
                                if self.download_cancelled:
                                    f.close()
                                    if os.path.exists(file_path):
                                        try:
                                            os.remove(file_path)
                                        except:
                                            pass
                                    file_response.close()
                                    session.close()
                                    self.download_response = None
                                    self.download_session = None
                                    # Hapus direktori yang tidak lengkap
                                    if os.path.exists(model_dir_path):
                                        import shutil
                                        try:
                                            shutil.rmtree(model_dir_path)
                                            print(f"Removed incomplete directory: {model_dir_path}")
                                        except:
                                            pass
                                    if model_name in self.downloading_models:
                                        del self.downloading_models[model_name]
                                    self.download_success.emit(model_name, False, "Download cancelled by user")
                                    return
                                
                                if chunk:
                                    f.write(chunk)
                                    file_downloaded += len(chunk)
                                    downloaded_total += len(chunk)
                                    
                                    # Update progress
                                    if total_size > 0:
                                        progress = min(int((downloaded_total / total_size) * 100), 99)
                                    else:
                                        progress = 0
                                    self.download_progress_update.emit(progress)
                                    self.download_size_update.emit(downloaded_total, total_size)
                        
                        file_response.close()
                        print(f"✓ Downloaded {filename}")
                    
                    # Buat refs/main/HEAD untuk struktur Hugging Face
                    refs_dir = os.path.join(model_dir_path, "refs", "main")
                    os.makedirs(refs_dir, exist_ok=True)
                    with open(os.path.join(refs_dir, "HEAD"), 'w') as f:
                        f.write(snapshot_id)
                    
                    # Close session
                    session.close()
                    self.download_session = None
                    self.download_response = None
                    
                    # Verify download
                    if os.path.exists(snapshot_path):
                        actual_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk(snapshot_path)
                            for filename in filenames
                        )
                        self.download_progress_update.emit(100)
                        self.download_size_update.emit(actual_size, actual_size)
                        print(f"✓ Faster Whisper model '{model_name}' downloaded successfully ({self.format_file_size(actual_size)})")
                    else:
                        raise Exception("Model directory not found after download")
                    
                    if model_name in self.downloading_models:
                        del self.downloading_models[model_name]
                    self.download_success.emit(model_name, True, "")
                    
                except Exception as e:
                    error_msg = str(e)
                    is_cancelled = "cancelled" in error_msg.lower() or self.download_cancelled
                    if not is_cancelled:
                        print(f"Error downloading Faster Whisper model: {error_msg}")
                        import traceback
                        traceback.print_exc()
                    if model_name in self.downloading_models:
                        del self.downloading_models[model_name]
                    # Cleanup session
                    if hasattr(self, 'download_session') and self.download_session:
                        try:
                            self.download_session.close()
                        except:
                            pass
                        self.download_session = None
                    if hasattr(self, 'download_response') and self.download_response:
                        try:
                            self.download_response.close()
                        except:
                            pass
                        self.download_response = None
                    # Jika cancel, cleanup file yang sudah didownload
                    if is_cancelled:
                        model_dir_path_cleanup = os.path.join(self.get_model_cache_path(), f"models--guillaumekln--faster-whisper-{model_name}")
                        if os.path.exists(model_dir_path_cleanup):
                            import shutil
                            try:
                                shutil.rmtree(model_dir_path_cleanup)
                                print(f"Cleaned up cancelled download for {model_name}")
                            except Exception as cleanup_error:
                                print(f"Warning: Could not clean up cancelled download: {cleanup_error}")
                    self.download_success.emit(model_name, False, error_msg if not is_cancelled else "Download cancelled by user")
            
            self.download_thread = threading.Thread(target=download_faster_whisper_thread, daemon=True)
            self.download_thread.start()
            return
        
        # Download model di thread terpisah (untuk OpenAI Whisper)
        self.download_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.download_size_label.setVisible(True)
        self.download_size_label.setText("Preparing download...")
        self.cancel_btn.setVisible(True)
        
        def download_thread():
            try:
                print(f"Downloading model: {model_name}")
                
                cache_dir = self.get_model_cache_path()
                os.makedirs(cache_dir, exist_ok=True)
                model_path = self.get_model_file_path(model_name)
                
                # Hapus file lama jika ada (untuk re-download)
                if os.path.exists(model_path):
                    try:
                        if os.path.isdir(model_path):
                            import shutil
                            shutil.rmtree(model_path)
                        else:
                            os.remove(model_path)
                    except:
                        pass
                
                # Gunakan whisper untuk mendapatkan URL download
                # Whisper menggunakan _MODELS dari whisper/__init__.py
                # Kita akan gunakan approach: download via whisper tapi dengan kontrol cancel
                import time
                
                # Estimasi ukuran model
                estimated_sizes = {
                    "tiny": 75 * 1024 * 1024,
                    "base": 150 * 1024 * 1024,
                    "small": 500 * 1024 * 1024,
                    "medium": 1500 * 1024 * 1024,
                    "large": 3000 * 1024 * 1024
                }
                estimated_size = estimated_sizes.get(model_name, 500 * 1024 * 1024)
                
                self.download_progress_update.emit(0)
                self.download_size_update.emit(0, estimated_size)
                
                # Download langsung menggunakan requests dengan stream=True
                # Ini memungkinkan kita untuk benar-benar menghentikan download saat cancel
                
                # Dapatkan URL model dari whisper
                try:
                    model_url = whisper._MODELS.get(model_name)
                    if not model_url:
                        raise Exception(f"Model '{model_name}' not found in Whisper models")
                except AttributeError:
                    # Fallback: gunakan whisper.load_model() jika _MODELS tidak tersedia
                    raise Exception("Could not get model URL. Please use whisper.load_model() directly.")
                
                print(f"Downloading {model_name} from: {model_url}")
                
                # Buat session untuk download yang bisa dihentikan
                session = requests.Session()
                self.download_session = session
                
                # Start download dengan stream=True
                response = session.get(model_url, stream=True, timeout=30)
                response.raise_for_status()
                self.download_response = response
                
                # Dapatkan total size dari header
                total_size = int(response.headers.get('content-length', estimated_size))
                self.download_size_update.emit(0, total_size)
                
                # Download dengan chunk dan check cancel flag
                downloaded = 0
                chunk_size = 8192  # 8KB chunks
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        # Check cancel flag setiap chunk
                        if self.download_cancelled:
                            print(f"Cancelling download for {model_name}...")
                            f.close()  # Close file
                            # Hapus file yang tidak lengkap
                            if os.path.exists(model_path):
                                try:
                                    os.remove(model_path)
                                    print(f"Removed incomplete file: {model_path}")
                                except:
                                    pass
                            # Close response
                            response.close()
                            session.close()
                            self.download_response = None
                            self.download_session = None
                            # Cleanup
                            if model_name in self.downloading_models:
                                del self.downloading_models[model_name]
                            self.download_success.emit(model_name, False, "Download cancelled by user")
                            return
                        
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            progress = min(int((downloaded / total_size) * 100), 99)
                            self.download_progress_update.emit(progress)
                            self.download_size_update.emit(downloaded, total_size)
                
                # Close response dan session
                response.close()
                session.close()
                self.download_response = None
                self.download_session = None
                
                # Verify file
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    final_size = os.path.getsize(model_path)
                    self.download_progress_update.emit(100)
                    self.download_size_update.emit(final_size, final_size)
                    if model_name in self.downloading_models:
                        del self.downloading_models[model_name]
                    self.download_success.emit(model_name, True, "")
                else:
                    raise Exception("Model file not found or empty after download")
                    
            except Exception as e:
                error_msg = str(e)
                print(f"Error downloading model: {error_msg}")
                import traceback
                traceback.print_exc()
                if model_name in self.downloading_models:
                    del self.downloading_models[model_name]
                self.download_success.emit(model_name, False, error_msg)
            finally:
                # Cleanup session
                if self.download_session:
                    try:
                        self.download_session.close()
                    except:
                        pass
                    self.download_session = None
        
        self.download_thread = threading.Thread(target=download_thread, daemon=True)
        self.download_thread.start()
    
    def update_download_progress(self, value):
        """Update progress bar (thread-safe via signal)"""
        self.download_progress.setValue(value)
    
    def update_download_size(self, downloaded_bytes, total_bytes):
        """Update label progress size (thread-safe via signal)"""
        downloaded_str = self.format_file_size(downloaded_bytes)
        total_str = self.format_file_size(total_bytes)
        self.download_size_label.setText(f"{downloaded_str} / {total_str}")
    
    def cancel_download(self):
        """Cancel download yang sedang berlangsung"""
        reply = QMessageBox.question(
            self,
            self.tr("cancel_download"),
            self.tr("cancel_confirmation"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.download_cancelled = True
            self.cancel_btn.setEnabled(False)
            self.download_size_label.setText("Cancelling...")
            
            # Hentikan response dan session download jika ada (untuk OpenAI Whisper)
            if hasattr(self, 'download_response') and self.download_response:
                try:
                    self.download_response.close()
                    print("Closed download response to stop download")
                except:
                    pass
                self.download_response = None
            
            # Hentikan session download (untuk OpenAI Whisper dan Faster Whisper)
            if hasattr(self, 'download_session') and self.download_session:
                try:
                    self.download_session.close()
                    print("Closed download session")
                except:
                    pass
                self.download_session = None
            
            # Cari model yang sedang didownload dan hapus filenya
            for model_name in list(self.downloading_models.keys()):
                if self.downloading_models[model_name]:
                    model_path = self.get_model_file_path(model_name)
                    # Hapus file yang tidak lengkap
                    if os.path.exists(model_path):
                        try:
                            # Untuk Faster Whisper, model_path adalah direktori
                            # Untuk OpenAI Whisper, model_path adalah file
                            if os.path.isdir(model_path):
                                import shutil
                                shutil.rmtree(model_path)
                                print(f"Cancelled download: Removed incomplete directory {model_path}")
                            else:
                                os.remove(model_path)
                                print(f"Cancelled download: Removed incomplete file {model_path}")
                        except Exception as e:
                            print(f"Warning: Could not remove file/directory: {e}")
                    break
    
    def on_download_complete(self, model_name, success, error_msg=""):
        """Callback untuk update UI setelah download selesai"""
        self.download_btn.setEnabled(True)
        self.download_progress.setVisible(False)
        self.download_progress.setValue(0)
        self.download_size_label.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self.download_cancelled = False
        
        # Unmark downloading (jika masih ada)
        if model_name in self.downloading_models:
            del self.downloading_models[model_name]
        
        # Refresh tabel untuk update status
        self.refresh_model_list()
        
        if success:
            QMessageBox.information(self, self.tr("success"), self.tr("model_downloaded", model_name=model_name))
        else:
            if "cancelled" not in error_msg.lower():
                QMessageBox.critical(self, self.tr("error"), self.tr("download_failed", model_name=model_name, error_msg=error_msg))
            # Jika cancelled, tidak perlu show error message
    
    def delete_model(self, model_name):
        """Hapus model yang sudah terdownload"""
        reply = QMessageBox.question(
            self,
            self.tr("confirm_delete"),
            f"{self.tr('delete_confirmation', model_name=model_name)}\n\n{self.tr('delete_warning')}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            model_path = self.get_model_file_path(model_name)
            try:
                # Handle Faster Whisper (directory) vs OpenAI Whisper (file)
                if self.engine == "faster" and os.path.isdir(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                    QMessageBox.information(self, self.tr("success"), self.tr("model_deleted", model_name=model_name))
                    self.refresh_model_list()
                elif os.path.exists(model_path):
                    os.remove(model_path)
                    QMessageBox.information(self, self.tr("success"), self.tr("model_deleted", model_name=model_name))
                    self.refresh_model_list()
                else:
                    QMessageBox.warning(self, self.tr("warning"), self.tr("model_not_found", model_path=model_path))
            except Exception as e:
                QMessageBox.critical(self, self.tr("error"), self.tr("delete_failed", error_msg=str(e)))


class BlackBorderLabel(QLabel):
    """Custom QLabel dengan black border font (outline text)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.use_black_border = False
        self.font_color = QColor(0, 0, 0)  # Default black (visible on white background)
    
    def paintEvent(self, event):
        """Override paintEvent untuk menggambar text dengan black border"""
        if self.use_black_border:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Ambil text dan font
            text = self.text()
            font = self.font()
            painter.setFont(font)
            
            # Hitung text rect
            text_rect = self.contentsRect()
            flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
            
            # Gambar black border (outline) dengan menggambar text beberapa kali di posisi berbeda
            painter.setPen(QPen(QColor(0, 0, 0, 255), 3))  # Black outline, 3px width
            
            # Gambar outline di 8 arah (untuk efek border yang halus)
            offsets = [
                (-2, -2), (-2, 0), (-2, 2),
                (0, -2), (0, 2),
                (2, -2), (2, 0), (2, 2)
            ]
            
            for dx, dy in offsets:
                painter.drawText(text_rect.adjusted(dx, dy, dx, dy), flags, text)
            
            # Gambar text dengan warna yang dipilih di tengah
            painter.setPen(self.font_color)
            painter.drawText(text_rect, flags, text)
        else:
            # Normal rendering
            super().paintEvent(event)


class FloatingCaptionWindow(QWidget):
    """Floating window untuk menampilkan caption overlay (seperti Windows Live Captions)"""
    def __init__(self, parent=None):
        # Set parent=None untuk membuat window benar-benar terpisah
        super().__init__(None)  # Tidak ada parent, window independent
        # Gunakan window decor normal tapi tetap always on top
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Window  # Window dengan decor normal
        )
        
        self.min_width = 300
        self.min_height = 60
        
        # Settings state
        self.always_on_top = True  # Default on
        self.black_border_font = False
        self.font_size = 14
        self.font_color = QColor(0, 0, 0)  # Default black
        
        # Set window title
        self.setWindowTitle("Caption Overlay")
        
        # Setup UI
        self.setup_ui()
        
        # Position di tengah bawah layar
        self.center_bottom()
        self.setMinimumSize(self.min_width, self.min_height)
    
    def setup_ui(self):
        """Setup UI untuk floating caption window"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container dengan background putih - tanpa border yang aneh
        self.container = QWidget()
        self.container.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(15, 12, 15, 12)
        
        # Label untuk caption text (gunakan custom label untuk black border)
        self.caption_label = BlackBorderLabel()
        self.caption_label.setWordWrap(True)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setFont(QFont("Arial", self.font_size, QFont.Weight.Normal))
        self.caption_label.setText("")
        # Set font color ke default (black) agar terlihat di white background
        self.caption_label.font_color = self.font_color
        container_layout.addWidget(self.caption_label)
        
        layout.addWidget(self.container)
        self.setLayout(layout)
        
        # Settings button di pojok kanan atas (overlay di container)
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(240, 240, 240, 200);
                border: 1px solid #666;
                border-radius: 15px;
                font-size: 16px;
                color: #333;
            }
            QPushButton:hover {
                background-color: rgba(220, 220, 220, 250);
            }
        """)
        self.settings_btn.clicked.connect(self.open_settings)
        self.settings_btn.setParent(self.container)
        self.settings_btn.raise_()  # Pastikan di atas
        
        # Store reference to parent MainWindow untuk notifikasi saat window ditutup
        self.parent_main_window = None
    
    def showEvent(self, event):
        """Override showEvent untuk memposisikan settings button saat window ditampilkan"""
        super().showEvent(event)
        # Delay sedikit untuk memastikan container sudah di-render
        QTimer.singleShot(10, lambda: self.resizeEvent(None))
    
    def center_bottom(self):
        """Posisikan window di tengah bawah layar"""
        screen = QApplication.primaryScreen().geometry()
        window_width = 600
        window_height = 100
        x = (screen.width() - window_width) // 2
        y = screen.height() - window_height - 50  # 50px dari bawah
        self.setGeometry(x, y, window_width, window_height)
    
    def update_caption_style(self):
        """Update style caption label berdasarkan settings"""
        self.caption_label.setFont(QFont("Arial", self.font_size, QFont.Weight.Normal))
        self.caption_label.use_black_border = self.black_border_font
        self.caption_label.font_color = self.font_color  # Store font color for paintEvent
        
        if self.black_border_font:
            # Black border font - akan di-handle oleh custom paintEvent
            self.caption_label.setStyleSheet(f"""
                QLabel {{
                    background-color: transparent;
                    padding: 5px;
                }}
            """)
        else:
            # Normal text dengan warna yang dipilih
            color_name = self.font_color.name()
            self.caption_label.setStyleSheet(f"""
                QLabel {{
                    color: {color_name};
                    background-color: transparent;
                    padding: 5px;
                }}
            """)
    
    def resizeEvent(self, event):
        """Override resizeEvent untuk memposisikan settings button"""
        super().resizeEvent(event)
        if hasattr(self, 'settings_btn') and hasattr(self, 'container'):
            # Posisikan di pojok kanan atas container
            btn_size = self.settings_btn.size()
            container_rect = self.container.geometry()
            x = container_rect.width() - btn_size.width() - 5
            y = 5  # Pojok kanan atas
            self.settings_btn.move(x, y)
    
    def update_caption(self, text):
        """Update caption text"""
        if text:
            # Pastikan font color sudah di-sync
            self.caption_label.font_color = self.font_color
            self.caption_label.setText(text)
            # Force update/repaint
            self.caption_label.update()
            self.container.update()
            # Window size tetap, tidak auto-resize
        else:
            self.caption_label.setText("")
            self.caption_label.update()
    
    def open_settings(self):
        """Buka dialog settings"""
        dialog = CaptionSettingsDialog(self)
        if dialog.exec():
            # Apply settings
            self.always_on_top = dialog.always_on_top_checkbox.isChecked()
            self.black_border_font = dialog.black_border_checkbox.isChecked()
            self.font_size = dialog.font_size_slider.value()
            self.font_color = dialog.font_color
            
            # Update always on top
            if not IS_WAYLAND:  # Hanya untuk Windows dan X11
                if self.always_on_top:
                    self.setWindowFlags(
                        Qt.WindowType.WindowStaysOnTopHint |
                        Qt.WindowType.Window
                    )
                else:
                    self.setWindowFlags(Qt.WindowType.Window)
                self.show()  # Re-show untuk apply flags
                self.raise_()
                self.activateWindow()
            
            # Update font style
            self.update_caption_style()
            
            # Update caption text untuk apply style baru
            current_text = self.caption_label.text()
            if current_text:
                self.update_caption(current_text)
    
    def closeEvent(self, event):
        """Override closeEvent untuk notifikasi parent saat window ditutup"""
        # Notifikasi parent MainWindow bahwa window ditutup
        if hasattr(self, 'parent_main_window') and self.parent_main_window:
            self.parent_main_window.on_floating_window_closed()
        super().closeEvent(event)


class CaptionSettingsDialog(QDialog):
    """Dialog untuk settings floating caption window"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Caption Settings")
        self.setModal(True)
        self.resize(350, 280)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        # Always on top checkbox (hanya untuk Windows dan X11)
        self.always_on_top_checkbox = QCheckBox("Always on Top")
        self.always_on_top_checkbox.setChecked(parent.always_on_top if parent else True)
        # Sembunyikan jika Wayland
        if IS_WAYLAND:
            self.always_on_top_checkbox.setVisible(False)
        else:
            form_layout.addRow(self.always_on_top_checkbox)
        
        # Black border font checkbox
        self.black_border_checkbox = QCheckBox("Black Border Font")
        self.black_border_checkbox.setChecked(parent.black_border_font if parent else False)
        form_layout.addRow(self.black_border_checkbox)
        
        # Font color button
        font_color_label = QLabel("Font Color:")
        self.font_color_btn = QPushButton()
        self.font_color = parent.font_color if parent else QColor(0, 0, 0)
        self.update_font_color_button()
        self.font_color_btn.clicked.connect(self.choose_font_color)
        form_layout.addRow(font_color_label, self.font_color_btn)
        
        # Font size slider
        font_size_label = QLabel("Font Size:")
        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setMinimum(8)
        self.font_size_slider.setMaximum(48)
        self.font_size_slider.setValue(parent.font_size if parent else 14)
        self.font_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_size_slider.setTickInterval(4)
        
        self.font_size_value_label = QLabel(str(self.font_size_slider.value()))
        self.font_size_slider.valueChanged.connect(
            lambda v: self.font_size_value_label.setText(str(v))
        )
        
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(self.font_size_slider)
        font_size_layout.addWidget(self.font_size_value_label)
        form_layout.addRow(font_size_label, font_size_layout)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
    
    def update_font_color_button(self):
        """Update tampilan button font color"""
        color_name = self.font_color.name()
        self.font_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color_name};
                border: 2px solid #666;
                border-radius: 4px;
                min-width: 80px;
                min-height: 25px;
            }}
        """)
        self.font_color_btn.setText(self.font_color.name())
    
    def choose_font_color(self):
        """Buka color picker untuk memilih font color"""
        color = QColorDialog.getColor(self.font_color, self, "Choose Font Color")
        if color.isValid():
            self.font_color = color
            self.update_font_color_button()


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
        
        # UI Language (default: English)
        self.ui_language = "en"
        
        # Translation dictionary
        self.translations = {
            "en": {
                "app_title": "Auto Captioning Application",
                "settings": "Settings",
                "stt_engine": "STT Engine:",
                "whisper_model": "Model:",
                "language": "Language:",
                "ui_language": "UI Language:",
                "model_manager": "Model Manager",
                "video_file_processing": "Video File Processing",
                "no_file_selected": "No file selected",
                "browse_video_file": "Browse Video File",
                "process_video": "Process Video",
                "system_audio_capture": "System Audio Capture (Real-time)",
                "start_system_capture": "Start System Audio Capture",
                "stop_system_capture": "Stop System Audio Capture",
                "clear_captions": "Clear Captions",
                "show_floating_overlay": "Show Floating Window",
                "captions": "Captions",
                "export_text_file": "Export as Text File",
                "export_srt": "Export as SRT (Subtitle)",
                "loading_video": "Loading video file...",
                "extracting_audio": "Extracting audio from video...",
                "audio_extracted": "Audio extracted. Loading Whisper model...",
                "loading_model": "Loading Whisper model...",
                "transcribing": "Transcribing audio...",
                "processing_completed": "Processing completed!",
                "error": "Error",
                "success": "Success",
                "video_processing_completed": "Video processing completed!",
                "please_select_file": "Please select a video file first",
                "video_no_audio": "Video has no audio track",
                "failed_extract_audio": "Failed to extract audio",
                "failed_load_model": "Failed to load Whisper model",
                "no_captions_export": "No captions to export",
                "captions_exported": "Captions exported to",
                "srt_exported": "SRT file exported to",
                "failed_export": "Failed to export",
                "failed_start_capture": "Failed to start audio capture",
                "pipewire_pulse_running": "Make sure PipeWire or PulseAudio is running.",
                "wasapi_available": "Make sure WASAPI is available. You may need to enable 'Stereo Mix' in Windows sound settings.",
                "check_audio_config": "Check your audio system configuration."
            },
            "id": {
                "app_title": "Aplikasi Auto Captioning",
                "settings": "Pengaturan",
                "stt_engine": "Engine STT:",
                "whisper_model": "Model:",
                "language": "Bahasa:",
                "ui_language": "Bahasa Antarmuka:",
                "model_manager": "Pengelola Model",
                "video_file_processing": "Pemrosesan File Video",
                "no_file_selected": "Tidak ada file dipilih",
                "browse_video_file": "Pilih File Video",
                "process_video": "Proses Video",
                "system_audio_capture": "Capture Audio Sistem (Real-time)",
                "start_system_capture": "Mulai Capture Audio Sistem",
                "stop_system_capture": "Hentikan Capture Audio Sistem",
                "clear_captions": "Hapus Caption",
                "show_floating_overlay": "Tampilkan Window Mengambang",
                "captions": "Caption",
                "export_text_file": "Ekspor sebagai File Teks",
                "export_srt": "Ekspor sebagai SRT (Subtitle)",
                "loading_video": "Memuat file video...",
                "extracting_audio": "Mengekstrak audio dari video...",
                "audio_extracted": "Audio diekstrak. Memuat model Whisper...",
                "loading_model": "Memuat model Whisper...",
                "transcribing": "Mentranskripsi audio...",
                "processing_completed": "Pemrosesan selesai!",
                "error": "Kesalahan",
                "success": "Berhasil",
                "video_processing_completed": "Pemrosesan video selesai!",
                "please_select_file": "Silakan pilih file video terlebih dahulu",
                "video_no_audio": "Video tidak memiliki track audio",
                "failed_extract_audio": "Gagal mengekstrak audio",
                "failed_load_model": "Gagal memuat model Whisper",
                "no_captions_export": "Tidak ada caption untuk diekspor",
                "captions_exported": "Caption diekspor ke",
                "srt_exported": "File SRT diekspor ke",
                "failed_export": "Gagal mengekspor",
                "failed_start_capture": "Gagal memulai capture audio",
                "pipewire_pulse_running": "Pastikan PipeWire atau PulseAudio berjalan.",
                "wasapi_available": "Pastikan WASAPI tersedia. Anda mungkin perlu mengaktifkan 'Stereo Mix' di pengaturan suara Windows.",
                "check_audio_config": "Periksa konfigurasi sistem audio Anda."
            }
        }
        
        # Setup UI
        self.setup_ui()
        
        # Timer untuk proses real-time
        self.realtime_timer = QTimer()
        self.realtime_timer.timeout.connect(self.process_realtime_chunk)
        self.audio_buffer = []
        self.buffer_duration = 3.0  # Buffer 3 detik
        
        # Floating caption window
        self.floating_caption_window = None
    
    def tr(self, key):
        """Mendapatkan terjemahan untuk key tertentu"""
        return self.translations.get(self.ui_language, self.translations["en"]).get(key, key)
    
    def set_ui_language(self, language_code):
        """Mengubah bahasa UI aplikasi"""
        if language_code in ["en", "id"]:
            self.ui_language = language_code
            self.update_ui_language()
    
    def update_ui_language(self):
        """Update semua teks UI berdasarkan bahasa yang dipilih"""
        # Window title
        self.setWindowTitle(self.tr("app_title"))
        
        # Title label
        if hasattr(self, 'title_label'):
            self.title_label.setText(self.tr("app_title"))
        
        # Settings group
        if hasattr(self, 'settings_group'):
            self.settings_group.setTitle(self.tr("settings"))
        
        # Settings labels
        if hasattr(self, 'model_label'):
            self.model_label.setText(self.tr("whisper_model"))
        if hasattr(self, 'language_label'):
            self.language_label.setText(self.tr("language"))
        if hasattr(self, 'ui_language_label'):
            self.ui_language_label.setText(self.tr("ui_language"))
        
        # Model Manager button
        if hasattr(self, 'model_manager_btn'):
            self.model_manager_btn.setText(self.tr("model_manager"))
        
        # Video File Processing
        if hasattr(self, 'video_group'):
            self.video_group.setTitle(self.tr("video_file_processing"))
        if hasattr(self, 'file_label'):
            if not hasattr(self, 'video_path'):
                self.file_label.setText(self.tr("no_file_selected"))
        if hasattr(self, 'browse_btn'):
            self.browse_btn.setText(self.tr("browse_video_file"))
        if hasattr(self, 'process_btn'):
            self.process_btn.setText(self.tr("process_video"))
        
        # System Audio Capture
        if hasattr(self, 'system_group'):
            self.system_group.setTitle(self.tr("system_audio_capture"))
        if hasattr(self, 'capture_btn'):
            if self.is_capturing:
                self.capture_btn.setText(self.tr("stop_system_capture"))
            else:
                self.capture_btn.setText(self.tr("start_system_capture"))
        if hasattr(self, 'clear_btn'):
            self.clear_btn.setText(self.tr("clear_captions"))
        if hasattr(self, 'overlay_checkbox'):
            self.overlay_checkbox.setText(self.tr("show_floating_overlay"))
        
        # Captions
        if hasattr(self, 'caption_group'):
            self.caption_group.setTitle(self.tr("captions"))
        if hasattr(self, 'export_txt_btn'):
            self.export_txt_btn.setText(self.tr("export_text_file"))
        if hasattr(self, 'export_srt_btn'):
            self.export_srt_btn.setText(self.tr("export_srt"))
    
    def on_ui_language_changed(self, language_text):
        """Handler saat UI language berubah"""
        if language_text == "English":
            self.set_ui_language("en")
        elif language_text == "Bahasa Indonesia":
            self.set_ui_language("id")

    def setup_ui(self):
        """Menyiapkan antarmuka pengguna"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Judul
        self.title_label = QLabel(self.tr("app_title"))
        self.title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Grup Settings
        self.settings_group = QGroupBox(self.tr("settings"))
        settings_layout = QVBoxLayout()
        
        # Baris pertama: STT Engine, Model, Language, dan UI Language
        first_row = QHBoxLayout()
        
        # Pemilihan STT Engine
        self.engine_label = QLabel(self.tr("stt_engine"))
        first_row.addWidget(self.engine_label)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["OpenAI Whisper", "Faster Whisper"])
        self.engine_combo.setCurrentText("OpenAI Whisper")
        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        first_row.addWidget(self.engine_combo)
        
        first_row.addSpacing(20)
        
        # Pemilihan model
        self.model_label = QLabel(self.tr("whisper_model"))
        first_row.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        first_row.addWidget(self.model_combo)
        
        first_row.addSpacing(20)  # Spacing antara model dan language
        
        # Pemilihan bahasa untuk captioning
        self.language_label = QLabel(self.tr("language"))
        first_row.addWidget(self.language_label)
        self.language_combo = QComboBox()
        # Format: "Display Name (code)"
        self.language_combo.addItems([
            "English (en)",
            "Bahasa Indonesia (id)"
        ])
        self.language_combo.setCurrentText("English (en)")
        first_row.addWidget(self.language_combo)
        
        first_row.addSpacing(20)  # Spacing antara language dan UI language
        
        # Pemilihan bahasa UI
        self.ui_language_label = QLabel(self.tr("ui_language"))
        first_row.addWidget(self.ui_language_label)
        self.ui_language_combo = QComboBox()
        self.ui_language_combo.addItems([
            "English",
            "Bahasa Indonesia"
        ])
        self.ui_language_combo.setCurrentText("English")
        self.ui_language_combo.currentTextChanged.connect(self.on_ui_language_changed)
        first_row.addWidget(self.ui_language_combo)
        
        first_row.addSpacing(20)  # Spacing antara UI language dan model manager
        
        # Tombol Model Manager
        self.model_manager_btn = QPushButton(self.tr("model_manager"))
        self.model_manager_btn.clicked.connect(self.open_model_manager)
        first_row.addWidget(self.model_manager_btn)
        
        first_row.addStretch()
        settings_layout.addLayout(first_row)
        
        self.settings_group.setLayout(settings_layout)
        layout.addWidget(self.settings_group)
        
        # Grup Pemrosesan File Video
        self.video_group = QGroupBox(self.tr("video_file_processing"))
        video_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.file_label = QLabel(self.tr("no_file_selected"))
        file_layout.addWidget(self.file_label)
        
        self.browse_btn = QPushButton(self.tr("browse_video_file"))
        self.browse_btn.clicked.connect(self.browse_video_file)
        file_layout.addWidget(self.browse_btn)
        
        self.process_btn = QPushButton(self.tr("process_video"))
        self.process_btn.clicked.connect(self.process_video_file)
        self.process_btn.setEnabled(False)
        file_layout.addWidget(self.process_btn)
        
        video_layout.addLayout(file_layout)
        
        # Progress bar untuk video processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setFormat("%p% - %v/%m")
        video_layout.addWidget(self.progress_bar)
        
        # Label status untuk video processing
        self.video_status_label = QLabel()
        self.video_status_label.setVisible(False)
        self.video_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_status_label)
        
        self.video_group.setLayout(video_layout)
        layout.addWidget(self.video_group)
        
        # Grup Capture Audio Sistem
        self.system_group = QGroupBox(self.tr("system_audio_capture"))
        system_layout = QVBoxLayout()
        
        control_layout = QHBoxLayout()
        self.capture_btn = QPushButton(self.tr("start_system_capture"))
        self.capture_btn.clicked.connect(self.toggle_system_capture)
        control_layout.addWidget(self.capture_btn)
        
        self.clear_btn = QPushButton(self.tr("clear_captions"))
        self.clear_btn.clicked.connect(self.clear_captions)
        control_layout.addWidget(self.clear_btn)
        
        system_layout.addLayout(control_layout)
        
        # Checkbox untuk floating overlay window
        self.overlay_checkbox = QCheckBox(self.tr("show_floating_overlay"))
        self.overlay_checkbox.setChecked(False)
        self.overlay_checkbox.toggled.connect(self.toggle_floating_overlay)
        system_layout.addWidget(self.overlay_checkbox)
        
        self.system_group.setLayout(system_layout)
        layout.addWidget(self.system_group)
        
        # Tampilan Caption
        self.caption_group = QGroupBox(self.tr("captions"))
        caption_layout = QVBoxLayout()
        
        self.caption_display = QTextEdit()
        self.caption_display.setReadOnly(True)
        self.caption_display.setFont(QFont("Arial", 12))
        caption_layout.addWidget(self.caption_display)
        
        # Tombol export
        export_layout = QHBoxLayout()
        self.export_txt_btn = QPushButton(self.tr("export_text_file"))
        self.export_txt_btn.clicked.connect(self.export_text_file)
        export_layout.addWidget(self.export_txt_btn)
        
        self.export_srt_btn = QPushButton(self.tr("export_srt"))
        self.export_srt_btn.clicked.connect(self.export_srt_file)
        export_layout.addWidget(self.export_srt_btn)
        
        caption_layout.addLayout(export_layout)
        self.caption_group.setLayout(caption_layout)
        layout.addWidget(self.caption_group)
        
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
            QMessageBox.warning(self, self.tr("error"), self.tr("please_select_file"))
            return
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.video_status_label.setVisible(True)
        self.video_status_label.setText(self.tr("loading_video"))
        
        # Ekstrak audio dari video
        audio_path = None
        try:
            # Update progress: Loading video (10%)
            self.progress_bar.setValue(10)
            self.video_status_label.setText(self.tr("loading_video"))
            
            video = VideoFileClip(self.video_path)
            
            if video.audio is None:
                raise Exception(self.tr("video_no_audio"))
            
            # Update progress: Video loaded (20%)
            self.progress_bar.setValue(20)
            self.video_status_label.setText(self.tr("extracting_audio"))
            
            # Buat path untuk file audio temporary
            audio_path = os.path.join(
                os.path.dirname(self.video_path) if os.path.dirname(self.video_path) else os.getcwd(),
                f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            
            print(f"Extracting audio to: {audio_path}")
            
            # Extract audio dengan format yang kompatibel dengan Whisper
            # MoviePy akan extract sebagai WAV, Whisper akan handle resampling otomatis
            # Update progress saat extracting (30-50%)
            self.progress_bar.setValue(30)
            video.audio.write_audiofile(audio_path)
            self.progress_bar.setValue(50)
            
            video.close()
            
            # Validasi file audio yang diekstrak
            if not os.path.exists(audio_path):
                raise Exception("Audio file was not created")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise Exception("Extracted audio file is empty")
            
            print(f"Audio extracted successfully: {file_size} bytes")
            
            # Update progress: Audio extracted (60%)
            self.progress_bar.setValue(60)
            self.video_status_label.setText(self.tr("audio_extracted"))
            
        except Exception as e:
            error_msg = f"{self.tr('failed_extract_audio')}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, self.tr("error"), error_msg)
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.video_status_label.setVisible(False)
            # Cleanup jika file dibuat tapi error
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            return
        
        # Proses audio dengan Whisper
        language_code = self.get_language_code()
        engine = self.get_stt_engine()
        self.caption_worker = CaptionWorker(
            self.model_combo.currentText(),
            language=language_code,
            engine=engine
        )
        
        # Update progress untuk loading model (60-70%)
        self.progress_bar.setValue(60)
        self.video_status_label.setText(self.tr("loading_model"))
        
        self.worker_thread = threading.Thread(
            target=self.caption_worker.process_audio_file,
            args=(audio_path,)
        )
        
        # Connect signals dengan progress mapping
        # Whisper progress (0-100) akan di-map ke 70-100% dari total progress
        def update_progress_with_status(whisper_progress):
            # Map whisper progress (0-100) ke range 70-100 dari total progress
            total_progress = 70 + int(whisper_progress * 0.3)
            self.progress_bar.setValue(total_progress)
            self.video_status_label.setText(f"{self.tr('transcribing')} {whisper_progress}%")
        
        self.caption_worker.caption_ready.connect(self.add_caption)
        self.caption_worker.progress_update.connect(update_progress_with_status)
        self.caption_worker.finished.connect(self.on_processing_finished)
        self.caption_worker.error.connect(self.on_processing_error)
        
        self.worker_thread.start()
        
        # Bersihkan file audio sementara setelah proses
        def cleanup():
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        self.caption_worker.finished.connect(cleanup)

    def get_stt_engine(self):
        """Mendapatkan STT engine yang dipilih"""
        engine_text = self.engine_combo.currentText()
        if "Faster" in engine_text:
            return "faster"
        return "openai"
    
    def on_engine_changed(self, engine_text):
        """Handler saat engine berubah"""
        if "Faster" in engine_text:
            self.current_stt_engine = "faster"
            # Update model list untuk Faster Whisper (sama seperti OpenAI Whisper)
            # Models tetap sama: tiny, base, small, medium, large
        else:
            self.current_stt_engine = "openai"
    
    def get_language_code(self):
        """Mengambil kode bahasa dari combobox"""
        language_text = self.language_combo.currentText()
        # Format: "English (en)" -> extract "en"
        if "(" in language_text and ")" in language_text:
            return language_text.split("(")[1].split(")")[0]
        return "en"  # Default ke English
    
    def open_model_manager(self):
        """Membuka dialog Model Manager"""
        engine = self.get_stt_engine()
        dialog = ModelManagerDialog(self, engine=engine)
        # Connect signal untuk update language saat berubah
        if hasattr(self, 'ui_language_combo'):
            self.ui_language_combo.currentTextChanged.connect(lambda: dialog.update_ui_language())
        dialog.exec()
        # Refresh model combo setelah dialog ditutup (jika ada model baru)
        # Model combo sudah memiliki semua model, jadi tidak perlu refresh
    
    def toggle_system_capture(self):
        """Toggle capture audio sistem"""
        if not self.is_capturing:
            self.start_system_capture()
        else:
            self.stop_system_capture()

    def start_system_capture(self):
        """Memulai capture audio sistem"""
        self.is_capturing = True
        self.capture_btn.setText(self.tr("stop_system_capture"))
        self.capture_btn.setStyleSheet("background-color: #ff4444;")
        
        # Inisialisasi worker dengan language yang dipilih
        language_code = self.get_language_code()
        engine = self.get_stt_engine()
        self.caption_worker = CaptionWorker(
            self.model_combo.currentText(),
            language=language_code,
            engine=engine
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
                self.tr("error"),
                f"{self.tr('failed_start_capture')}\n{platform_msg}"
            )
            self.stop_system_capture()

    def stop_system_capture(self):
        """Menghentikan capture audio sistem"""
        self.is_capturing = False
        self.capture_btn.setText(self.tr("start_system_capture"))
        self.capture_btn.setStyleSheet("")
        
        if self.realtime_timer.isActive():
            self.realtime_timer.stop()
        
        if self.audio_capture:
            self.audio_capture.stop_capture()
        
        if self.caption_worker:
            self.caption_worker.stop()
        
        # Clear floating overlay saat stop
        if self.floating_caption_window:
            self.floating_caption_window.update_caption("")

    def process_realtime_chunk(self):
        """Memproses buffer audio yang terkumpul"""
        if not self.audio_buffer:
            return
        
        if not self.caption_worker:
            return
        
        # Pastikan model sudah dimuat
        if not self.caption_worker.model:
            # Jangan spam print, hanya print sekali setiap beberapa detik
            if not hasattr(self, '_last_model_wait_print'):
                self._last_model_wait_print = 0
            import time
            current_time = time.time()
            if current_time - self._last_model_wait_print > 5.0:  # Print setiap 5 detik
                print("Model not loaded yet, waiting... (This may take a while if downloading for the first time)")
                self._last_model_wait_print = current_time
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
        
        # Debug: print caption yang diterima
        print(f"add_caption called: text='{text}', start={start_time:.2f}, end={end_time:.2f}")
        
        # Simpan caption
        self.captions.append({
            "text": text,
            "start": start_time,
            "end": end_time,
            "timestamp": timestamp
        })
        
        # Perbarui tampilan utama
        self.caption_display.append(f"{timestamp} {time_str}: {text}")
        
        # Auto-scroll ke bawah
        cursor = self.caption_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.caption_display.setTextCursor(cursor)
        
        # Update floating overlay window jika ada
        if self.floating_caption_window:
            print(f"Floating window exists: visible={self.floating_caption_window.isVisible()}")
            if self.floating_caption_window.isVisible():
                print(f"Updating floating window with text: '{text}'")
                self.floating_caption_window.update_caption(text)
            else:
                print("Floating window is not visible")
        else:
            print("Floating window is None")

    def clear_captions(self):
        """Menghapus semua caption"""
        self.caption_display.clear()
        self.captions = []
        
        # Clear floating overlay juga
        if self.floating_caption_window:
            self.floating_caption_window.update_caption("")
    
    def toggle_floating_overlay(self, checked):
        """Toggle floating overlay window"""
        if checked:
            if self.floating_caption_window is None:
                # Jangan set parent agar window benar-benar terpisah
                self.floating_caption_window = FloatingCaptionWindow(None)
                # Set reference ke MainWindow untuk notifikasi saat window ditutup
                self.floating_caption_window.parent_main_window = self
            self.floating_caption_window.show()
            self.floating_caption_window.raise_()  # Pastikan di atas
            self.floating_caption_window.activateWindow()  # Aktifkan window
        else:
            if self.floating_caption_window:
                self.floating_caption_window.hide()
    
    def on_floating_window_closed(self):
        """Callback saat floating window ditutup via title bar close button"""
        # Uncheck checkbox dan clear reference
        if hasattr(self, 'overlay_checkbox'):
            self.overlay_checkbox.blockSignals(True)  # Block signals untuk mencegah loop
            self.overlay_checkbox.setChecked(False)
            self.overlay_checkbox.blockSignals(False)
        self.floating_caption_window = None

    def export_text_file(self):
        """Ekspor caption sebagai file teks biasa"""
        if not self.captions:
            QMessageBox.warning(self, self.tr("error"), self.tr("no_captions_export"))
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("export_text_file"),
            f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for caption in self.captions:
                        f.write(f"[{caption['start']:.2f}s - {caption['end']:.2f}s] {caption['text']}\n")
                QMessageBox.information(self, self.tr("success"), f"{self.tr('captions_exported')} {file_path}")
            except Exception as e:
                QMessageBox.critical(self, self.tr("error"), f"{self.tr('failed_export')}: {str(e)}")

    def export_srt_file(self):
        """Ekspor caption sebagai file subtitle SRT"""
        if not self.captions:
            QMessageBox.warning(self, self.tr("error"), self.tr("no_captions_export"))
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("export_srt"),
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
                QMessageBox.information(self, self.tr("success"), f"{self.tr('srt_exported')} {file_path}")
            except Exception as e:
                QMessageBox.critical(self, self.tr("error"), f"{self.tr('failed_export')}: {str(e)}")

    def format_srt_time(self, seconds):
        """Format detik ke format waktu SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def on_processing_finished(self):
        """Dipanggil ketika pemrosesan video selesai"""
        self.progress_bar.setValue(100)
        self.video_status_label.setText("Processing completed!")
        self.process_btn.setEnabled(True)
        # Sembunyikan progress bar setelah beberapa detik
        QTimer.singleShot(2000, lambda: (
            self.progress_bar.setVisible(False),
            self.video_status_label.setVisible(False)
        ))
        QMessageBox.information(self, "Success", "Video processing completed!")

    def on_processing_error(self, error_msg):
        """Menangani error pemrosesan"""
        QMessageBox.critical(self, self.tr("error"), error_msg)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.video_status_label.setVisible(False)
        if self.is_capturing:
            self.stop_system_capture()

    def closeEvent(self, event):
        """Menangani event penutupan jendela"""
        if self.is_capturing:
            self.stop_system_capture()
        if self.caption_worker:
            self.caption_worker.stop()
        
        # Tutup floating overlay window jika ada
        if hasattr(self, 'floating_caption_window') and self.floating_caption_window:
            self.floating_caption_window.close()
            self.floating_caption_window = None
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
