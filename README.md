# Auto Captioning Application

A comprehensive application for automatic captioning of audio from video files and real-time system-wide audio capture. Perfect for transcribing videos, creating subtitles, and captioning system audio (useful for gaming, streaming, etc.).

## Features

- **Video File Processing**: Extract audio from video files and generate captions with progress tracking
- **Real-time System Audio Capture**: Capture and caption system-wide audio in real-time
- **Multiple STT Engines**: Choose between OpenAI Whisper and Faster Whisper for different speed/accuracy trade-offs
- **Model Manager**: Download and manage different STT engines and models
- **Floating Caption Window**: Separate overlay window for real-time captions (similar to Windows Live Captions)
  - Always on top (Windows/Linux X11)
  - Customizable font size, color, and black border effect
  - Draggable and resizable
- **Multiple Export Formats**: Export captions as plain text (.txt) or SRT subtitle files
- **Multiple Whisper Models**: Choose from tiny, base, small, medium, or large models for different accuracy/speed trade-offs
- **Language Support**: 
  - Transcription languages: English, Bahasa Indonesia
  - UI languages: English, Bahasa Indonesia
- **Progress Tracking**: Real-time progress bars for video processing and model downloads
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
- You can choose between OpenAI Whisper and Faster Whisper in the application settings.
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
2. Select STT Engine (OpenAI Whisper or Faster Whisper)
3. Select a Whisper model (base is recommended for good balance of speed and accuracy)
4. Select transcription language (English or Bahasa Indonesia)
5. Click "Process Video" to extract audio and generate captions
6. Progress bar will show the processing status
7. Captions will appear in real-time in the caption display area
8. Export captions as text or SRT format using the export buttons

### System Audio Capture

1. Select STT Engine, model, and language in the settings
2. Click "Start System Audio Capture" to begin capturing system-wide audio
3. The application will capture all system audio (games, videos, music, etc.)
4. Captions will appear in real-time as speech is detected
5. Enable "Show Floating Window" to display captions in a separate overlay window
6. Click "Stop System Audio Capture" to stop
7. Export captions when finished

### Model Manager

1. Click "Model Manager" button to open the model management dialog
2. View all available models and their download status
3. Download different STT engines and models as needed
4. Models are managed separately for OpenAI Whisper and Faster Whisper engines

### Floating Caption Window

1. Enable "Show Floating Window" checkbox in System Audio Capture section
2. A separate window will appear for displaying real-time captions
3. Click the settings (⚙) button in the floating window to customize:
   - Always on top (Windows/Linux X11 only, not available on Wayland)
   - Black border font effect
   - Font size (8-48pt)
   - Font color
4. Drag and resize the window as needed
5. The window will automatically close when you stop audio capture or close the main window

### Export Options

- **Export as Text File**: Creates a plain text file with timestamps and captions
- **Export as SRT File**: Creates a standard SRT subtitle file compatible with video players

## STT Engine Selection

### OpenAI Whisper
- Original OpenAI Whisper implementation
- Models stored in `~/.cache/whisper/` (Linux) or `%LOCALAPPDATA%\whisper\` (Windows)
- File format: `.pt` files
- **Pros**: Well-tested, stable
- **Cons**: Slower inference speed

### Faster Whisper
- Optimized implementation using CTranslate2
- Models stored in `~/.cache/huggingface/hub/` (Linux) or `%LOCALAPPDATA%\huggingface\hub\` (Windows)
- Directory-based storage
- **Pros**: 2-4x faster inference, lower latency
- **Cons**: Requires cuDNN for CUDA acceleration (falls back to CPU if unavailable)

**Recommendation**: Use Faster Whisper for real-time captioning to reduce latency.

## Model Selection

- **tiny**: Fastest, least accurate (~39M parameters)
- **base**: Good balance (~74M parameters) - **Recommended**
- **small**: Better accuracy (~244M parameters)
- **medium**: High accuracy (~769M parameters)
- **large**: Best accuracy, slowest (~1550M parameters)

**Note**: Model sizes are the same for both OpenAI Whisper and Faster Whisper, but Faster Whisper runs faster.

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

- **Speech Recognition**: 
  - OpenAI Whisper (original implementation)
  - Faster Whisper (optimized implementation with CTranslate2)
- **Audio Processing**: MoviePy, SoundDevice
- **System Audio Capture**:
  - **Linux**: PipeWire (preferred) or PulseAudio (via pulsectl)
  - **Windows**: WASAPI loopback (via pycaw)
- **GUI Framework**: PyQt6
- **Cross-Platform**: Automatically detects OS and uses appropriate audio backend
- **Model Management**: 
  - OpenAI Whisper: Direct download from OpenAI CDN
  - Faster Whisper: Download from Hugging Face Hub

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

## Notes

- **⚠️ IMPORTANT: Always use virtual environment!** This application must be run within a virtual environment to avoid dependency conflicts with your system Python.
- First run will download the Whisper model (can be several hundred MB)
- Real-time captioning has a slight delay (2-3 seconds with OpenAI Whisper, 1-2 seconds with Faster Whisper) due to processing time
- **Linux**: Works with both PipeWire (Wayland) and PulseAudio (X11)
- **Windows**: Uses WASAPI for system audio capture
- For best results, use Faster Whisper with "base" or "small" model for real-time captioning
- The application automatically detects your platform and uses the appropriate audio backend
- Faster Whisper will automatically fallback to CPU if CUDA/cuDNN is not available
- Floating window "Always on Top" feature is not available on Wayland (Linux) due to platform limitations


- **Each users or developers that wants to test it must create their own virtual environment** by running `setup.sh` or `setup.bat`
- **Always activate virtual environment** before running the application
- If there are issues with dependencies, delete the `venv/` folder and run the setup script again

---

# Aplikasi Auto Captioning

Aplikasi komprehensif untuk captioning otomatis audio dari file video dan capture audio sistem secara real-time. Sempurna untuk mentranskripsi video, membuat subtitle, dan captioning audio sistem (berguna untuk gaming, streaming, dll.).

## Fitur

- **Pemrosesan File Video**: Ekstrak audio dari file video dan hasilkan caption dengan progress tracking
- **Capture Audio Sistem Real-time**: Tangkap dan caption audio sistem secara real-time
- **Multiple STT Engine**: Pilih antara OpenAI Whisper dan Faster Whisper untuk trade-off kecepatan/akurasi yang berbeda
- **Model Manager**: Download dan kelola berbagai STT engine dan model
- **Floating Caption Window**: Window overlay terpisah untuk caption real-time (mirip Windows Live Captions)
  - Always on top (Windows/Linux X11)
  - Font size, warna, dan efek black border dapat disesuaikan
  - Dapat digeser dan diubah ukurannya
- **Format Export Beragam**: Ekspor caption sebagai file teks (.txt) atau subtitle SRT
- **Model Whisper Beragam**: Pilih dari model tiny, base, small, medium, atau large untuk trade-off akurasi/kecepatan yang berbeda
- **Dukungan Bahasa**: 
  - Bahasa transkripsi: English, Bahasa Indonesia
  - Bahasa UI: English, Bahasa Indonesia
- **Progress Tracking**: Progress bar real-time untuk pemrosesan video dan download model
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
- Anda dapat memilih antara OpenAI Whisper dan Faster Whisper di pengaturan aplikasi.
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
2. Pilih STT Engine (OpenAI Whisper atau Faster Whisper)
3. Pilih model Whisper (base direkomendasikan untuk keseimbangan kecepatan dan akurasi)
4. Pilih bahasa transkripsi (English atau Bahasa Indonesia)
5. Klik "Process Video" untuk mengekstrak audio dan menghasilkan caption
6. Progress bar akan menampilkan status pemrosesan
7. Caption akan muncul secara real-time di area tampilan caption
8. Ekspor caption sebagai teks atau format SRT menggunakan tombol export

### Capture Audio Sistem

1. Pilih STT Engine, model, dan bahasa di pengaturan
2. Klik "Start System Audio Capture" untuk mulai menangkap audio sistem
3. Aplikasi akan menangkap semua audio sistem (game, video, musik, dll.)
4. Caption akan muncul secara real-time saat speech terdeteksi
5. Aktifkan "Show Floating Window" untuk menampilkan caption di window overlay terpisah
6. Klik "Stop System Audio Capture" untuk berhenti
7. Ekspor caption setelah selesai

### Model Manager

1. Klik tombol "Model Manager" untuk membuka dialog manajemen model
2. Lihat semua model yang tersedia dan status download mereka
3. Download berbagai STT engine dan model sesuai kebutuhan
4. Model dikelola terpisah untuk engine OpenAI Whisper dan Faster Whisper

### Floating Caption Window

1. Aktifkan checkbox "Show Floating Window" di bagian System Audio Capture
2. Window terpisah akan muncul untuk menampilkan caption real-time
3. Klik tombol settings (⚙) di floating window untuk menyesuaikan:
   - Always on top (Windows/Linux X11 saja, tidak tersedia di Wayland)
   - Efek black border font
   - Ukuran font (8-48pt)
   - Warna font
4. Geser dan ubah ukuran window sesuai kebutuhan
5. Window akan otomatis tertutup saat Anda menghentikan audio capture atau menutup window utama

### Opsi Export

- **Export as Text File**: Membuat file teks biasa dengan timestamp dan caption
- **Export as SRT File**: Membuat file subtitle SRT standar yang kompatibel dengan video player

## Pemilihan STT Engine

### OpenAI Whisper
- Implementasi OpenAI Whisper asli
- Model disimpan di `~/.cache/whisper/` (Linux) atau `%LOCALAPPDATA%\whisper\` (Windows)
- Format file: file `.pt`
- **Kelebihan**: Teruji, stabil
- **Kekurangan**: Kecepatan inference lebih lambat

### Faster Whisper
- Implementasi yang dioptimalkan menggunakan CTranslate2
- Model disimpan di `~/.cache/huggingface/hub/` (Linux) atau `%LOCALAPPDATA%\huggingface\hub\` (Windows)
- Penyimpanan berbasis direktori
- **Kelebihan**: 2-4x lebih cepat, latency lebih rendah
- **Kekurangan**: Membutuhkan cuDNN untuk akselerasi CUDA (fallback ke CPU jika tidak tersedia)

**Rekomendasi**: Gunakan Faster Whisper untuk captioning real-time untuk mengurangi latency.

## Pemilihan Model

- **tiny**: Tercepat, paling tidak akurat (~39M parameter)
- **base**: Keseimbangan baik (~74M parameter) - **Direkomendasikan**
- **small**: Akurasi lebih baik (~244M parameter)
- **medium**: Akurasi tinggi (~769M parameter)
- **large**: Akurasi terbaik, paling lambat (~1550M parameter)

**Catatan**: Ukuran model sama untuk OpenAI Whisper dan Faster Whisper, tapi Faster Whisper berjalan lebih cepat.

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

- **Pengenalan Suara**: 
  - OpenAI Whisper (implementasi asli)
  - Faster Whisper (implementasi yang dioptimalkan dengan CTranslate2)
- **Pemrosesan Audio**: MoviePy, SoundDevice
- **Capture Audio Sistem**:
  - **Linux**: PipeWire (preferensi) atau PulseAudio (via pulsectl)
  - **Windows**: WASAPI loopback (via pycaw)
- **Framework GUI**: PyQt6
- **Cross-Platform**: Otomatis mendeteksi OS dan menggunakan audio backend yang sesuai
- **Manajemen Model**: 
  - OpenAI Whisper: Download langsung dari OpenAI CDN
  - Faster Whisper: Download dari Hugging Face Hub

## Lisensi

Proyek ini dilisensikan di bawah MIT License.

Lihat file [LICENSE](LICENSE) untuk detail.

## Catatan

- **⚠️ PENTING: Selalu gunakan virtual environment!** Aplikasi ini harus dijalankan di dalam virtual environment untuk menghindari konflik dependencies dengan sistem Python Anda.
- Pertama kali menjalankan akan mengunduh model Whisper (bisa beberapa ratus MB)
- Captioning real-time memiliki delay sedikit (2-3 detik dengan OpenAI Whisper, 1-2 detik dengan Faster Whisper) karena waktu pemrosesan
- **Linux**: Bekerja dengan PipeWire (Wayland) dan PulseAudio (X11)
- **Windows**: Menggunakan WASAPI untuk capture audio sistem
- Untuk hasil terbaik, gunakan Faster Whisper dengan model "base" atau "small" untuk captioning real-time
- Aplikasi otomatis mendeteksi platform Anda dan menggunakan audio backend yang sesuai
- Faster Whisper akan otomatis fallback ke CPU jika CUDA/cuDNN tidak tersedia
- Fitur "Always on Top" pada floating window tidak tersedia di Wayland (Linux) karena keterbatasan platform


- **Setiap user & developer yang ingin mengetes aplikasi ini harus membuat virtual environment sendiri** dengan menjalankan `setup.sh` atau `setup.bat`
- **Selalu aktifkan virtual environment** sebelum menjalankan aplikasi
- Jika ada masalah dengan dependencies, hapus folder `venv/` dan jalankan setup script lagi
