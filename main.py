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
from moviepy.editor import VideoFileClip

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

    def __init__(self, model_name="base"):
        super().__init__()
        self.model = None
        self.model_name = model_name
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
        
        self.is_running = True
        try:
            result = self.model.transcribe(
                audio_path,
                language="en",  # Dapat dibuat konfigurasi
                task="transcribe",
                verbose=False
            )
            
            for segment in result["segments"]:
                if not self.is_running:
                    break
                self.caption_ready.emit(
                    segment["text"].strip(),
                    segment["start"],
                    segment["end"]
                )
                self.progress_update.emit(int((segment["end"] / result["segments"][-1]["end"]) * 100))
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Error processing audio: {str(e)}")
        finally:
            self.is_running = False

    def process_realtime_audio(self, audio_data):
        """Memproses chunk audio real-time"""
        if not self.model:
            self.load_model()
        
        try:
            # Konversi ke float32 dan normalisasi
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            result = self.model.transcribe(
                audio_float,
                language="en",
                task="transcribe",
                verbose=False,
                fp16=False
            )
            
            if result["segments"]:
                for segment in result["segments"]:
                    if segment["text"].strip():
                        self.caption_ready.emit(
                            segment["text"].strip(),
                            segment["start"],
                            segment["end"]
                        )
        except Exception as e:
            self.error.emit(f"Error processing real-time audio: {str(e)}")

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
        
        # Coba PulseAudio/PipeWire via pulsectl jika tersedia
        if PULSE_AVAILABLE:
            try:
                self.pulse = pulsectl.Pulse('audio-capture')
                sources = self.pulse.source_list()
                monitor_source = None
                
                # Cari source .monitor (PipeWire/PulseAudio)
                for source in sources:
                    if source.name.endswith('.monitor'):
                        monitor_source = source.name
                        break
                
                if not monitor_source:
                    # Coba cari default monitor
                    try:
                        default_source = self.pulse.server_info().default_source_name
                        for source in sources:
                            if default_source in source.name or source.name.endswith('.monitor'):
                                monitor_source = source.name
                                break
                    except:
                        pass
                
                # Cari device yang cocok di sounddevice
                if monitor_source:
                    devices = sd.query_devices()
                    for i, device in enumerate(devices):
                        device_name_lower = device['name'].lower()
                        if (monitor_source in device['name'] or 
                            'monitor' in device_name_lower or
                            'pipewire' in device_name_lower or
                            'pulse' in device_name_lower):
                            device_id = i
                            break
            except Exception as e:
                print(f"Error deteksi PulseAudio/PipeWire: {str(e)}")
        
        # Fallback: cari device monitor di sounddevice
        if device_id is None:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name_lower = device['name'].lower()
                if ('monitor' in device_name_lower or 
                    'pipewire' in device_name_lower or
                    'loopback' in device_name_lower):
                    device_id = i
                    break
        
        return device_id

    def _find_windows_loopback_device(self):
        """Mencari device WASAPI loopback di Windows"""
        device_id = None
        
        if PYCAW_AVAILABLE:
            try:
                # Ambil default playback device
                devices = AudioUtilities.GetSpeakers()
                device_name = devices.FriendlyName if devices else None
                
                if device_name:
                    # Cari device yang cocok di sounddevice
                    sd_devices = sd.query_devices()
                    for i, device in enumerate(sd_devices):
                        # Device WASAPI loopback biasanya punya penamaan khusus
                        if ('loopback' in device['name'].lower() or 
                            device_name in device['name'] or
                            'wasapi' in device['name'].lower()):
                            # Cek apakah ini input device
                            if device['max_input_channels'] > 0:
                                device_id = i
                                break
            except Exception as e:
                print(f"Error deteksi WASAPI: {str(e)}")
        
        # Fallback: cari loopback di nama device
        if device_id is None:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name_lower = device['name'].lower()
                if ('loopback' in device_name_lower or 
                    'wasapi' in device_name_lower or
                    'stereo mix' in device_name_lower):
                    if device['max_input_channels'] > 0:
                        device_id = i
                        break
        
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
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Status audio: {status}")
                if self.is_recording and self.callback:
                    # Konversi ke int16
                    audio_int16 = (indata * 32767).astype(np.int16)
                    self.callback(audio_int16.flatten())
            
            # Gunakan device yang terdeteksi atau fallback ke default input
            if device_id is None:
                device_id = sd.default.device[0] if sd.default.device[0] is not None else None
                print(f"Peringatan: Menggunakan default input device. Capture audio sistem mungkin tidak bekerja.")
                print(f"Device yang tersedia:")
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        print(f"  [{i}] {device['name']} (input: {device['max_input_channels']})")
            
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
            
            if device_id is not None:
                device_info = sd.query_devices(device_id)
                print(f"Menangkap dari: {device_info['name']}")
            
            return True
        except Exception as e:
            print(f"Error memulai capture audio: {str(e)}")
            # Fallback akhir ke default input
            try:
                def audio_callback(indata, frames, time, status):
                    if status:
                        print(f"Status audio: {status}")
                    if self.is_recording and self.callback:
                        audio_int16 = (indata * 32767).astype(np.int16)
                        self.callback(audio_int16.flatten())
                
                self.audio_stream = sd.InputStream(
                    device=None,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=audio_callback,
                    blocksize=self.chunk_size,
                    dtype=np.float32
                )
                self.is_recording = True
                self.audio_stream.start()
                print("Menggunakan fallback: default input device")
                return True
            except Exception as e2:
                print(f"Fallback capture audio juga gagal: {str(e2)}")
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
        settings_layout = QHBoxLayout()
        
        # Pemilihan model
        settings_layout.addWidget(QLabel("Whisper Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        settings_layout.addWidget(self.model_combo)
        
        settings_layout.addStretch()
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
        try:
            video = VideoFileClip(self.video_path)
            audio_path = os.path.join(
                os.path.dirname(self.video_path),
                f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            )
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to extract audio: {str(e)}")
            self.process_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        
        # Proses audio
        self.caption_worker = CaptionWorker(self.model_combo.currentText())
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
        
        # Inisialisasi worker
        self.caption_worker = CaptionWorker(self.model_combo.currentText())
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
        if not self.audio_buffer or not self.caption_worker or not self.caption_worker.model:
            return
        
        # Gabungkan buffer audio
        if self.audio_buffer:
            audio_chunk = np.concatenate(self.audio_buffer)
            self.audio_buffer.clear()
            
            # Proses di thread background
            threading.Thread(
                target=self.caption_worker.process_realtime_audio,
                args=(audio_chunk,),
                daemon=True
            ).start()

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
