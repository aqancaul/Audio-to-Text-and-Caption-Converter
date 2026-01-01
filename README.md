# Auto Captioning Application

A comprehensive application for automatic captioning of audio from video files and real-time system-wide audio capture. Perfect for transcribing videos, creating subtitles, and captioning system audio (useful for gaming, streaming, etc.).

## Features

- **Video File Processing**: Extract audio from video files and generate captions
- **Real-time System Audio Capture**: Capture and caption system-wide audio in real-time
- **Multiple Export Formats**: Export captions as plain text (.txt) or SRT subtitle files
- **Multiple Whisper Models**: Choose from tiny, base, small, medium, or large models for different accuracy/speed trade-offs
- **Modern GUI**: User-friendly PyQt6 interface
- **Cross-Platform**: Works on Linux (PipeWire/PulseAudio) and Windows (WASAPI)

## Requirements

- Python 3.8 or higher
- FFmpeg (for video processing)
- **Linux**: PipeWire or PulseAudio (for system audio capture)
- **Windows**: Windows Audio Session API (WASAPI) - built-in

## Installation

### Quick Setup (Recommended)

**Use the automatic setup script that will create a virtual environment:**

```bash
# Linux
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

The script will automatically:
- Create Python virtual environment (`venv/`)
- Activate virtual environment
- Install all required dependencies

### Manual Setup with Virtual Environment

**IMPORTANT: This application MUST be run within a Python virtual environment to avoid dependency conflicts.**

#### Linux

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

#### Windows

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Note**: 
- The first time you run the application, Whisper will download the selected model. This may take a few minutes depending on your internet connection and the model size.
- Platform-specific dependencies (pulsectl for Linux, pycaw for Windows) are automatically installed based on your OS.
- **Always activate virtual environment before running the application!**

## Usage

### Running the Application

**IMPORTANT: Make sure virtual environment is activated!**

```bash
# Linux
source venv/bin/activate
python main.py

# Windows
venv\Scripts\activate
python main.py
```

**To deactivate virtual environment after finishing:**
```bash
# Linux/Windows
deactivate
```

## Compiling to Standalone Executable

The application can be compiled into a standalone executable that does not require Python to be installed.

### Linux (AppImage)

```bash
# Setup and compile at once
./setup.sh --compile

# Or if already setup previously
source venv/bin/activate
pip install pyinstaller
./setup.sh --compile
```

Compilation output will be in the `dist_appimage/` folder:
- `AutoCaptioning-x86_64.AppImage` - AppImage that can run on all Linux distributions
- Or `AutoCaptioning` - Portable executable if appimagetool is not available

**Note for AppImage:**
- If `appimagetool` is not installed, the script will create a portable executable
- To create AppImage, install appimagetool:
  ```bash
  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x appimagetool-x86_64.AppImage
  sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
  ```

### Windows (Portable EXE)

```bash
# Setup and compile at once
setup.bat --compile

# Or if already setup previously
venv\Scripts\activate
pip install pyinstaller
setup.bat --compile
```

Compilation output will be in the `dist_exe/` folder:
- `AutoCaptioning.exe` - Portable executable that can run on Windows without Python

**Note:**
- Executable file is quite large (~500MB-1GB) because it includes all dependencies including PyTorch
- First run of executable will download Whisper model (if not already present)
- Executable is portable and can be shared to other Windows computers

### Video File Processing

1. Click "Browse Video File" to select a video file (supports MP4, AVI, MOV, MKV, WebM, FLV)
2. Select a Whisper model (base is recommended for good balance of speed and accuracy)
3. Click "Process Video" to extract audio and generate captions
4. Captions will appear in real-time in the caption display area
5. Export captions as text or SRT format using the export buttons

### System Audio Capture

1. Click "Start System Audio Capture" to begin capturing system-wide audio
2. The application will capture all system audio (games, videos, music, etc.)
3. Captions will appear in real-time as speech is detected
4. Click "Stop System Audio Capture" to stop
5. Export captions when finished

### Export Options

- **Export as Text File**: Creates a plain text file with timestamps and captions
- **Export as SRT File**: Creates a standard SRT subtitle file compatible with video players

## Model Selection

- **tiny**: Fastest, least accurate (~39M parameters)
- **base**: Good balance (~74M parameters) - **Recommended**
- **small**: Better accuracy (~244M parameters)
- **medium**: High accuracy (~769M parameters)
- **large**: Best accuracy, slowest (~1550M parameters)

## Troubleshooting

### System Audio Capture Not Working

#### Linux (PipeWire/PulseAudio)

- **PipeWire**: Usually works automatically. Check if running: `systemctl --user status pipewire`
- **PulseAudio**: Ensure it's running: `pulseaudio --check` or `pulseaudio --start`
- Check available audio sources: `pactl list sources short` or `pw-cli list-objects`
- Look for devices with `.monitor` suffix (these are system audio monitors)
- If using PipeWire, it should automatically expose monitor sources

#### Windows

- Ensure WASAPI is available (built into Windows)
- Try enabling "Stereo Mix" in Windows sound settings (Recording tab)
- Check available audio devices in the application console output
- Some audio drivers may not support loopback - try updating audio drivers

### Video Processing Fails

- Ensure FFmpeg is installed: `ffmpeg -version`
- Check that the video file is not corrupted
- Try a different video format

### Out of Memory Errors

- Use a smaller Whisper model (tiny or base)
- Close other applications to free up RAM
- Process shorter video files

## Technical Details

- **Speech Recognition**: OpenAI Whisper
- **Audio Processing**: MoviePy, SoundDevice
- **System Audio Capture**:
  - **Linux**: PipeWire (preferred) or PulseAudio (via pulsectl)
  - **Windows**: WASAPI loopback (via pycaw)
- **GUI Framework**: PyQt6
- **Cross-Platform**: Automatically detects OS and uses appropriate audio backend

## License

This project is licensed under the GNU General Public License v2.0 (GPLv2).

See the [LICENSE](LICENSE) file for details.

## Notes

- **⚠️ IMPORTANT: Always use virtual environment!** This application must be run within a virtual environment to avoid dependency conflicts with your system Python.
- First run will download the Whisper model (can be several hundred MB)
- Real-time captioning has a slight delay (2-3 seconds) due to processing time
- **Linux**: Works with both PipeWire (Wayland) and PulseAudio (X11)
- **Windows**: Uses WASAPI for system audio capture
- For best results, use the "base" or "small" model for real-time captioning
- The application automatically detects your platform and uses the appropriate audio backend

## Virtual Environment Best Practices

- **Do not commit `venv/` folder to Git** - already ignored in `.gitignore`
- **Each developer must create their own virtual environment** by running `setup.sh` or `setup.bat`
- **Always activate virtual environment** before running the application
- If there are issues with dependencies, delete the `venv/` folder and run the setup script again

---

# Aplikasi Auto Captioning

Aplikasi komprehensif untuk captioning otomatis audio dari file video dan capture audio sistem secara real-time. Sempurna untuk mentranskripsi video, membuat subtitle, dan captioning audio sistem (berguna untuk gaming, streaming, dll.).

## Fitur

- **Pemrosesan File Video**: Ekstrak audio dari file video dan hasilkan caption
- **Capture Audio Sistem Real-time**: Tangkap dan caption audio sistem secara real-time
- **Format Export Beragam**: Ekspor caption sebagai file teks (.txt) atau subtitle SRT
- **Model Whisper Beragam**: Pilih dari model tiny, base, small, medium, atau large untuk trade-off akurasi/kecepatan yang berbeda
- **GUI Modern**: Antarmuka PyQt6 yang ramah pengguna
- **Cross-Platform**: Bekerja di Linux (PipeWire/PulseAudio) dan Windows (WASAPI)

## Persyaratan

- Python 3.8 atau lebih tinggi
- FFmpeg (untuk pemrosesan video)
- **Linux**: PipeWire atau PulseAudio (untuk capture audio sistem)
- **Windows**: Windows Audio Session API (WASAPI) - built-in

## Instalasi

### Setup Cepat (Disarankan)

**Gunakan setup script otomatis yang akan membuat virtual environment:**

```bash
# Linux
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

Script akan otomatis:
- Membuat virtual environment Python (`venv/`)
- Mengaktifkan virtual environment
- Menginstall semua dependencies yang diperlukan

### Setup Manual dengan Virtual Environment

**PENTING: Aplikasi ini HARUS dijalankan di dalam virtual environment Python untuk menghindari konflik dependencies.**

#### Linux

```bash
# 1. Buat virtual environment
python3 -m venv venv

# 2. Aktifkan virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

#### Windows

```bash
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan virtual environment
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Catatan**: 
- Pertama kali menjalankan aplikasi, Whisper akan mengunduh model yang dipilih. Ini mungkin memakan waktu beberapa menit tergantung koneksi internet dan ukuran model.
- Dependencies khusus platform (pulsectl untuk Linux, pycaw untuk Windows) otomatis terinstall berdasarkan OS Anda.
- **Selalu aktifkan virtual environment sebelum menjalankan aplikasi!**

## Penggunaan

### Menjalankan Aplikasi

**PENTING: Pastikan virtual environment sudah diaktifkan!**

```bash
# Linux
source venv/bin/activate
python main.py

# Windows
venv\Scripts\activate
python main.py
```

**Untuk deactivate virtual environment setelah selesai:**
```bash
# Linux/Windows
deactivate
```

## Kompilasi ke Executable Standalone

Aplikasi dapat dikompilasi menjadi executable standalone yang tidak memerlukan Python terinstall.

### Linux (AppImage)

```bash
# Setup dan compile sekaligus
./setup.sh --compile

# Atau jika sudah setup sebelumnya
source venv/bin/activate
pip install pyinstaller
./setup.sh --compile
```

Hasil kompilasi akan berada di folder `dist_appimage/`:
- `AutoCaptioning-x86_64.AppImage` - AppImage yang dapat dijalankan di semua distribusi Linux
- Atau `AutoCaptioning` - Portable executable jika appimagetool tidak tersedia

**Catatan untuk AppImage:**
- Jika `appimagetool` tidak terinstall, script akan membuat portable executable
- Untuk membuat AppImage, install appimagetool:
  ```bash
  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x appimagetool-x86_64.AppImage
  sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
  ```

### Windows (Portable EXE)

```bash
# Setup dan compile sekaligus
setup.bat --compile

# Atau jika sudah setup sebelumnya
venv\Scripts\activate
pip install pyinstaller
setup.bat --compile
```

Hasil kompilasi akan berada di folder `dist_exe/`:
- `AutoCaptioning.exe` - Portable executable yang dapat dijalankan di Windows tanpa Python

**Catatan:**
- File executable cukup besar (~500MB-1GB) karena termasuk semua dependencies termasuk PyTorch
- Executable pertama kali dijalankan akan download Whisper model (jika belum ada)
- Executable bersifat portable dan dapat dibagikan ke komputer Windows lain

### Pemrosesan File Video

1. Klik "Browse Video File" untuk memilih file video (mendukung MP4, AVI, MOV, MKV, WebM, FLV)
2. Pilih model Whisper (base direkomendasikan untuk keseimbangan kecepatan dan akurasi)
3. Klik "Process Video" untuk mengekstrak audio dan menghasilkan caption
4. Caption akan muncul secara real-time di area tampilan caption
5. Ekspor caption sebagai teks atau format SRT menggunakan tombol export

### Capture Audio Sistem

1. Klik "Start System Audio Capture" untuk mulai menangkap audio sistem
2. Aplikasi akan menangkap semua audio sistem (game, video, musik, dll.)
3. Caption akan muncul secara real-time saat speech terdeteksi
4. Klik "Stop System Audio Capture" untuk berhenti
5. Ekspor caption setelah selesai

### Opsi Export

- **Export as Text File**: Membuat file teks biasa dengan timestamp dan caption
- **Export as SRT File**: Membuat file subtitle SRT standar yang kompatibel dengan video player

## Pemilihan Model

- **tiny**: Tercepat, paling tidak akurat (~39M parameter)
- **base**: Keseimbangan baik (~74M parameter) - **Direkomendasikan**
- **small**: Akurasi lebih baik (~244M parameter)
- **medium**: Akurasi tinggi (~769M parameter)
- **large**: Akurasi terbaik, paling lambat (~1550M parameter)

## Troubleshooting

### Capture Audio Sistem Tidak Bekerja

#### Linux (PipeWire/PulseAudio)

- **PipeWire**: Biasanya bekerja otomatis. Cek apakah berjalan: `systemctl --user status pipewire`
- **PulseAudio**: Pastikan berjalan: `pulseaudio --check` atau `pulseaudio --start`
- Cek audio source yang tersedia: `pactl list sources short` atau `pw-cli list-objects`
- Cari device dengan suffix `.monitor` (ini adalah monitor audio sistem)
- Jika menggunakan PipeWire, seharusnya otomatis mengekspos monitor source

#### Windows

- Pastikan WASAPI tersedia (built-in Windows)
- Coba aktifkan "Stereo Mix" di pengaturan suara Windows (tab Recording)
- Cek device audio yang tersedia di output konsol aplikasi
- Beberapa driver audio mungkin tidak mendukung loopback - coba update driver audio

### Pemrosesan Video Gagal

- Pastikan FFmpeg terinstall: `ffmpeg -version`
- Cek bahwa file video tidak korup
- Coba format video yang berbeda

### Error Out of Memory

- Gunakan model Whisper yang lebih kecil (tiny atau base)
- Tutup aplikasi lain untuk membebaskan RAM
- Proses file video yang lebih pendek

## Detail Teknis

- **Pengenalan Suara**: OpenAI Whisper
- **Pemrosesan Audio**: MoviePy, SoundDevice
- **Capture Audio Sistem**:
  - **Linux**: PipeWire (preferensi) atau PulseAudio (via pulsectl)
  - **Windows**: WASAPI loopback (via pycaw)
- **Framework GUI**: PyQt6
- **Cross-Platform**: Otomatis mendeteksi OS dan menggunakan audio backend yang sesuai

## Lisensi

Proyek ini dilisensikan di bawah GNU General Public License v2.0 (GPLv2).

Lihat file [LICENSE](LICENSE) untuk detail.

## Catatan

- **⚠️ PENTING: Selalu gunakan virtual environment!** Aplikasi ini harus dijalankan di dalam virtual environment untuk menghindari konflik dependencies dengan sistem Python Anda.
- Pertama kali menjalankan akan mengunduh model Whisper (bisa beberapa ratus MB)
- Captioning real-time memiliki delay sedikit (2-3 detik) karena waktu pemrosesan
- **Linux**: Bekerja dengan PipeWire (Wayland) dan PulseAudio (X11)
- **Windows**: Menggunakan WASAPI untuk capture audio sistem
- Untuk hasil terbaik, gunakan model "base" atau "small" untuk captioning real-time
- Aplikasi otomatis mendeteksi platform Anda dan menggunakan audio backend yang sesuai

## Best Practices Virtual Environment

- **Jangan commit folder `venv/` ke Git** - sudah di-ignore di `.gitignore`
- **Setiap developer harus membuat virtual environment sendiri** dengan menjalankan `setup.sh` atau `setup.bat`
- **Selalu aktifkan virtual environment** sebelum menjalankan aplikasi
- Jika ada masalah dengan dependencies, hapus folder `venv/` dan jalankan setup script lagi
