#!/bin/bash

# Auto Captioning Application - Setup Script

# Locale: use Indonesian if LANG/LC_ALL starts with id
if [[ "${LANG,,}" == id* || "${LC_ALL,,}" == id* ]]; then
    _lang=id
else
    _lang=en
fi

# Cek argumen --compile
COMPILE_MODE=false
if [[ "$1" == "--compile" ]]; then
    COMPILE_MODE=true
fi

if [ "$_lang" = "id" ]; then
    echo "Aplikasi Auto Captioning - Script Setup"
else
    echo "Auto Captioning Application - Setup Script"
fi
echo "=========================================="
if [ "$COMPILE_MODE" = true ]; then
    [ "$_lang" = "id" ] && echo "Mode: KOMPILASI (AppImage)" || echo "Mode: COMPILE (AppImage)"
fi
echo ""

# Cek Python 3
if ! command -v python3 &> /dev/null; then
    [ "$_lang" = "id" ] && echo "Error: Python 3 tidak terinstall. Pasang Python 3.8 atau lebih tinggi." || echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
[ "$_lang" = "id" ] && echo "✓ Python 3 ditemukan: $(python3 --version)" || echo "✓ Python 3 found: $(python3 --version)"

# Cek pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    [ "$_lang" = "id" ] && echo "Error: pip tidak terinstall. Pasang pip." || echo "Error: pip is not installed. Please install pip."
    exit 1
fi
[ "$_lang" = "id" ] && echo "✓ pip ditemukan" || echo "✓ pip found"

# Cek FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    if [ "$_lang" = "id" ]; then
        echo "Peringatan: FFmpeg tidak terinstall. Pemrosesan video tidak akan bekerja."
        echo "Pasang FFmpeg: Arch/CachyOS: sudo pacman -S ffmpeg | Ubuntu: sudo apt-get install ffmpeg | Fedora: sudo dnf install ffmpeg"
    else
        echo "Warning: FFmpeg is not installed. Video processing will not work."
        echo "Install: Arch/CachyOS: sudo pacman -S ffmpeg | Ubuntu: sudo apt-get install ffmpeg | Fedora: sudo dnf install ffmpeg"
    fi
else
    [ "$_lang" = "id" ] && echo "✓ FFmpeg ditemukan: $(ffmpeg -version | head -n 1)" || echo "✓ FFmpeg found: $(ffmpeg -version | head -n 1)"
fi

# Cek audio (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v pipewire &> /dev/null; then
        if systemctl --user is-active --quiet pipewire 2>/dev/null || pgrep -x pipewire > /dev/null; then
            [ "$_lang" = "id" ] && echo "✓ PipeWire berjalan (Wayland)" || echo "✓ PipeWire running (Wayland)"
        else
            [ "$_lang" = "id" ] && echo "Peringatan: PipeWire terinstall tapi mungkin tidak berjalan." || echo "Warning: PipeWire installed but may not be running."
        fi
    elif command -v pulseaudio &> /dev/null; then
        if pulseaudio --check &> /dev/null; then
            [ "$_lang" = "id" ] && echo "✓ PulseAudio berjalan" || echo "✓ PulseAudio running"
        else
            [ "$_lang" = "id" ] && echo "Peringatan: PulseAudio tidak berjalan. Jalankan: pulseaudio --start" || echo "Warning: PulseAudio not running. Run: pulseaudio --start"
        fi
    else
        if [ "$_lang" = "id" ]; then
            echo "Peringatan: PipeWire atau PulseAudio tidak ditemukan. Pasang: sudo pacman -S pipewire pipewire-pulse (atau apt/dnf setara)."
        else
            echo "Warning: PipeWire or PulseAudio not found. Install e.g. sudo pacman -S pipewire pipewire-pulse."
        fi
    fi
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    [ "$_lang" = "id" ] && echo "✓ Windows terdeteksi - WASAPI untuk capture audio" || echo "✓ Windows detected - WASAPI for audio capture"
fi

echo ""
[ "$_lang" = "id" ] && echo "Membuat virtual environment Python..." || echo "Creating Python virtual environment..."
echo ""

if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        [ "$_lang" = "id" ] && echo "Error: Gagal membuat virtual environment." || echo "Error: Failed to create virtual environment."
        exit 1
    fi
    [ "$_lang" = "id" ] && echo "✓ Virtual environment dibuat" || echo "✓ Virtual environment created"
else
    [ "$_lang" = "id" ] && echo "✓ Virtual environment sudah ada" || echo "✓ Virtual environment already exists"
fi

echo ""
[ "$_lang" = "id" ] && echo "Mengaktifkan venv dan menginstall dependencies..." || echo "Activating venv and installing dependencies..."
echo ""

source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Check the messages above."
    deactivate 2>/dev/null
    exit 1
fi

# --- Compile mode ---
if [ "$COMPILE_MODE" = true ]; then
    echo ""
    [ "$_lang" = "id" ] && echo "==========================================" && echo "Membangun Application..." || echo "==========================================" && echo "Building Application..."
    echo "=========================================="
    echo ""

    pip install pyinstaller > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        [ "$_lang" = "id" ] && echo "Error: Gagal menginstall PyInstaller." || echo "Error: Failed to install PyInstaller."
        deactivate 2>/dev/null
        exit 1
    fi

    BUILD_DIR="build_appimage"
    rm -rf "$BUILD_DIR" 2>/dev/null
    mkdir -p "$BUILD_DIR"

    APPDIR="$BUILD_DIR/AutoCaptioning.AppDir"
    rm -rf "$APPDIR" 2>/dev/null
    mkdir -p "$APPDIR/usr/bin"
    mkdir -p "$APPDIR/usr/share/applications"
    mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

    [ "$_lang" = "id" ] && echo "Mengompilasi dengan PyInstaller..." || echo "Compiling with PyInstaller..."

    pyinstaller --name="AutoCaptioning" \
        --windowed \
        --add-data="requirements.txt:." \
        --hidden-import=whisper \
        --hidden-import=faster_whisper \
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
        --collect-all=faster_whisper \
        --copy-metadata=imageio \
        --noconfirm \
        main.py

    if [ $? -ne 0 ]; then
        echo "Error: PyInstaller build failed."
        deactivate 2>/dev/null
        exit 1
    fi

    if [ ! -d "dist/AutoCaptioning" ]; then
        [ "$_lang" = "id" ] && echo "Error: Build PyInstaller tidak ditemukan (dist/AutoCaptioning)." || echo "Error: PyInstaller build not found (dist/AutoCaptioning)."
        deactivate 2>/dev/null
        exit 1
    fi

    cp -r "dist/AutoCaptioning"/* "$APPDIR/usr/bin/"
    chmod +x "$APPDIR/usr/bin/AutoCaptioning"
    rm -rf dist build 2>/dev/null

    cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
exec "${HERE}/usr/bin/AutoCaptioning" "$@"
EOF
    chmod +x "$APPDIR/AppRun"

    cat > "$APPDIR/usr/share/applications/autocaptioning.desktop" << 'EOF'
[Desktop Entry]
Name=Auto Captioning Application
Comment=Auto captioning for audio and video files
Exec=AutoCaptioning
Icon=autocaptioning
Type=Application
Categories=AudioVideo;Audio;Video;
EOF
    cp "$APPDIR/usr/share/applications/autocaptioning.desktop" "$APPDIR/autocaptioning.desktop"

    ICON_PATH="$APPDIR/usr/share/icons/hicolor/256x256/apps/autocaptioning.png"
    if command -v convert &> /dev/null; then
        convert -size 256x256 xc:blue -pointsize 72 -fill white -gravity center -annotate +0+0 "AC" "$ICON_PATH" 2>/dev/null || true
    fi
    if [ -f "$ICON_PATH" ]; then
        cp "$ICON_PATH" "$APPDIR/autocaptioning.png"
    fi

    if command -v appimagetool &> /dev/null; then
        [ "$_lang" = "id" ] && echo "Membuat AppImage..." || echo "Creating AppImage..."
        appimagetool "$APPDIR" "AutoCaptioning-x86_64.AppImage"
        if [ $? -eq 0 ] && [ -f "AutoCaptioning-x86_64.AppImage" ]; then
            chmod +x "AutoCaptioning-x86_64.AppImage"
            echo ""
            [ "$_lang" = "id" ] && echo "✓ AppImage berhasil dibuat: AutoCaptioning-x86_64.AppImage" || echo "✓ AppImage created: AutoCaptioning-x86_64.AppImage"
            echo "  ./AutoCaptioning-x86_64.AppImage"
        else
            if [ "$_lang" = "id" ]; then
                echo "Peringatan: appimagetool gagal. AppDir siap di: $APPDIR"
                echo "Jalankan aplikasi: $APPDIR/AppRun atau $APPDIR/usr/bin/AutoCaptioning"
            else
                echo "Warning: appimagetool failed. AppDir is ready at: $APPDIR"
                echo "Run app: $APPDIR/AppRun or $APPDIR/usr/bin/AutoCaptioning"
            fi
        fi
    else
        if [ "$_lang" = "id" ]; then
            echo "Peringatan: appimagetool tidak ditemukan. AppDir siap di: $APPDIR"
            echo "Jalankan aplikasi: $APPDIR/AppRun"
            echo ""
            echo "Untuk membuat AppImage nanti:"
            echo "  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
            echo "  chmod +x appimagetool-x86_64.AppImage"
            echo "  ARCH=x86_64 ./appimagetool-x86_64.AppImage $APPDIR"
        else
            echo "Warning: appimagetool not found. AppDir is ready at: $APPDIR"
            echo "Run the app: $APPDIR/AppRun"
            echo ""
            echo "To create an AppImage later:"
            echo "  wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
            echo "  chmod +x appimagetool-x86_64.AppImage"
            echo "  ARCH=x86_64 ./appimagetool-x86_64.AppImage $APPDIR"
        fi
    fi

    echo ""
    [ "$_lang" = "id" ] && echo "Build selesai. Output: $APPDIR (dan AppImage jika appimagetool terinstall)." || echo "Build done. Output: $APPDIR (and AppImage if appimagetool was installed)."
    echo ""

else
    echo ""
    if [ "$_lang" = "id" ]; then
        echo "✓ Setup berhasil!"
        echo "=========================================="
        echo "PENTING: Virtual environment sedang aktif!"
        echo "=========================================="
        echo ""
        echo "Untuk menjalankan aplikasi:"
        echo "  1. Aktifkan: source venv/bin/activate"
        echo "  2. Jalankan: python main.py"
        echo "  3. Nonaktifkan: deactivate"
        echo ""
        echo "Untuk kompilasi (AppImage): ./setup.sh --compile"
        echo "Catatan: Run pertama mungkin akan mengunduh model Whisper."
        echo ""
    else
        echo "✓ Setup completed successfully!"
        echo "=========================================="
        echo "IMPORTANT: Virtual environment is active!"
        echo "=========================================="
        echo ""
        echo "To run the application:"
        echo "  1. Activate: source venv/bin/activate"
        echo "  2. Run: python main.py"
        echo "  3. Deactivate: deactivate"
        echo ""
        echo "To compile (AppImage): ./setup.sh --compile"
        echo "Note: First run may download the Whisper model."
        echo ""
    fi
fi
