#!/bin/bash

# Script Setup Aplikasi Auto Captioning

# Cek argumen --compile
COMPILE_MODE=false
if [[ "$1" == "--compile" ]]; then
    COMPILE_MODE=true
fi

echo "Auto Captioning Application - Setup Script"
echo "=========================================="
if [ "$COMPILE_MODE" = true ]; then
    echo "Mode: COMPILE (AppImage mode)"
fi
echo ""

# Cek apakah Python 3 terinstall
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 tidak terinstall. Silakan install Python 3.8 atau lebih tinggi."
    exit 1
fi

echo "✓ Python 3 ditemukan: $(python3 --version)"

# Cek apakah pip terinstall
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "Error: pip tidak terinstall. Silakan install pip."
    exit 1
fi

echo "✓ pip ditemukan"

# Cek apakah FFmpeg terinstall
if ! command -v ffmpeg &> /dev/null; then
    echo "Peringatan: FFmpeg tidak terinstall. Pemrosesan video tidak akan bekerja."
    echo "Silakan install FFmpeg:"
    echo "  Arch/CachyOS: sudo pacman -S ffmpeg"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  Fedora: sudo dnf install ffmpeg"
else
    echo "✓ FFmpeg ditemukan: $(ffmpeg -version | head -n 1)"
fi

# Cek sistem audio (hanya Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Cek PipeWire (preferensi di Wayland)
    if command -v pipewire &> /dev/null; then
        if systemctl --user is-active --quiet pipewire 2>/dev/null || pgrep -x pipewire > /dev/null; then
            echo "✓ PipeWire sedang berjalan (kompatibel Wayland)"
        else
            echo "Peringatan: PipeWire terinstall tapi mungkin tidak berjalan."
        fi
    # Cek PulseAudio (fallback)
    elif command -v pulseaudio &> /dev/null; then
        if pulseaudio --check &> /dev/null; then
            echo "✓ PulseAudio sedang berjalan"
        else
            echo "Peringatan: PulseAudio terinstall tapi tidak berjalan."
            echo "Jalankan dengan: pulseaudio --start"
        fi
    else
        echo "Peringatan: PipeWire atau PulseAudio tidak ditemukan. Capture audio sistem mungkin tidak bekerja."
        echo "Silakan install salah satunya:"
        echo "  Arch/CachyOS: sudo pacman -S pipewire pipewire-pulse"
        echo "  Ubuntu/Debian: sudo apt-get install pipewire pipewire-pulse"
        echo "  Fedora: sudo dnf install pipewire pipewire-pulse"
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "✓ Windows terdeteksi - WASAPI akan digunakan untuk capture audio sistem"
fi

echo ""
echo "Membuat virtual environment Python..."
echo ""

# Buat virtual environment jika belum ada
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Gagal membuat virtual environment."
        exit 1
    fi
    echo "✓ Virtual environment dibuat"
else
    echo "✓ Virtual environment sudah ada"
fi

echo ""
echo "Mengaktifkan virtual environment dan menginstall dependencies..."
echo ""

# Aktifkan virtual environment dan install dependencies
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Gagal mengaktifkan virtual environment."
    exit 1
fi

# Upgrade pip terlebih dahulu
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies Python
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to install dependencies. Please check the error messages above."
    deactivate 2>/dev/null
    exit 1
fi

# Jika mode compile, build AppImage
if [ "$COMPILE_MODE" = true ]; then
    echo ""
    echo "=========================================="
    echo "Membangun AppImage..."
    echo "=========================================="
    echo ""
    
    # Install PyInstaller
    echo "Menginstall PyInstaller..."
    pip install pyinstaller > /dev/null 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Error: Gagal menginstall PyInstaller."
        deactivate 2>/dev/null
        exit 1
    fi
    
    # Buat direktori build
    BUILD_DIR="build_appimage"
    DIST_DIR="dist_appimage"
    rm -rf "$BUILD_DIR" "$DIST_DIR" 2>/dev/null
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
    
    # Buat struktur AppDir
    APPDIR="$BUILD_DIR/AutoCaptioning.AppDir"
    rm -rf "$APPDIR" 2>/dev/null
    mkdir -p "$APPDIR/usr/bin"
    mkdir -p "$APPDIR/usr/share/applications"
    mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
    
    echo "Mengompilasi aplikasi dengan PyInstaller..."
    
    # Build dengan PyInstaller
    pyinstaller --name="AutoCaptioning" \
        --onefile \
        --windowed \
        --add-data="requirements.txt:." \
        --hidden-import=whisper \
        --hidden-import=PyQt6 \
        --hidden-import=PyQt6.QtCore \
        --hidden-import=PyQt6.QtGui \
        --hidden-import=PyQt6.QtWidgets \
        --hidden-import=sounddevice \
        --hidden-import=numpy \
        --hidden-import=torch \
        --hidden-import=torchaudio \
        --hidden-import=moviepy \
        --hidden-import=pulsectl \
        --collect-all=whisper \
        --collect-all=torch \
        --noconfirm \
        main.py
    
    if [ $? -ne 0 ]; then
        echo "Error: PyInstaller build failed."
        deactivate 2>/dev/null
        exit 1
    fi
    
    # Salin executable ke AppDir
    if [ -f "dist/AutoCaptioning" ]; then
        cp "dist/AutoCaptioning" "$APPDIR/usr/bin/"
        chmod +x "$APPDIR/usr/bin/AutoCaptioning"
    else
        echo "Error: Executable PyInstaller tidak ditemukan."
        deactivate 2>/dev/null
        exit 1
    fi
    
    # Buat script AppRun
    cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/AutoCaptioning" "$@"
EOF
    chmod +x "$APPDIR/AppRun"
    
    # Buat file .desktop
    cat > "$APPDIR/usr/share/applications/autocaptioning.desktop" << 'EOF'
[Desktop Entry]
Name=Auto Captioning Application
Comment=Auto captioning for audio and video files
Exec=AutoCaptioning
Icon=autocaptioning
Type=Application
Categories=AudioVideo;Audio;Video;
EOF
    
    # Buat icon (placeholder - user bisa mengganti)
    if [ ! -f "$APPDIR/usr/share/icons/hicolor/256x256/apps/autocaptioning.png" ]; then
        # Buat icon placeholder sederhana menggunakan ImageMagick jika tersedia, atau skip
        if command -v convert &> /dev/null; then
            convert -size 256x256 xc:blue -pointsize 72 -fill white -gravity center -annotate +0+0 "AC" "$APPDIR/usr/share/icons/hicolor/256x256/apps/autocaptioning.png" 2>/dev/null || true
        fi
    fi
    
    # Buat AppImage menggunakan appimagetool jika tersedia, atau berikan instruksi
    if command -v appimagetool &> /dev/null; then
        echo "Membuat AppImage dengan appimagetool..."
        appimagetool "$APPDIR" "$DIST_DIR/AutoCaptioning-x86_64.AppImage"
        
        if [ $? -eq 0 ] && [ -f "$DIST_DIR/AutoCaptioning-x86_64.AppImage" ]; then
            chmod +x "$DIST_DIR/AutoCaptioning-x86_64.AppImage"
            echo ""
            echo "✓ AppImage berhasil dibuat!"
            echo "  Lokasi: $DIST_DIR/AutoCaptioning-x86_64.AppImage"
            echo ""
            echo "Untuk membuat executable dan menjalankan:"
            echo "  chmod +x $DIST_DIR/AutoCaptioning-x86_64.AppImage"
            echo "  ./$DIST_DIR/AutoCaptioning-x86_64.AppImage"
        else
            echo "Peringatan: appimagetool gagal. Membuat portable executable sebagai gantinya..."
            cp "$APPDIR/usr/bin/AutoCaptioning" "$DIST_DIR/AutoCaptioning"
            echo "Portable executable dibuat: $DIST_DIR/AutoCaptioning"
        fi
    else
        echo "Peringatan: appimagetool tidak ditemukan. Membuat portable executable sebagai gantinya..."
        echo "Untuk membuat AppImage, install appimagetool:"
        echo "  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
        echo "  chmod +x appimagetool-x86_64.AppImage"
        echo "  sudo mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool"
        echo ""
        cp "$APPDIR/usr/bin/AutoCaptioning" "$DIST_DIR/AutoCaptioning"
        echo "Portable executable dibuat: $DIST_DIR/AutoCaptioning"
        echo "Anda bisa membuat AppImage secara manual nanti menggunakan appimagetool."
    fi
    
    # Bersihkan file build PyInstaller (simpan dist untuk sementara)
    echo ""
    echo "Build selesai! Output di: $DIST_DIR/"
    echo ""
    
else
    echo ""
    echo "✓ Setup completed successfully!"
    echo ""
    echo "=========================================="
    echo "IMPORTANT: Virtual environment is active!"
    echo "=========================================="
    echo ""
    echo "To run the application:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Run application: python main.py"
    echo "  3. Deactivate when done: deactivate"
    echo ""
    echo "To compile to AppImage:"
    echo "  ./setup.sh --compile"
    echo ""
    echo "Note: The first run will download the Whisper model (may take a few minutes)."
    echo ""
    echo "Virtual environment is currently active. You can run 'python main.py' now."
fi
