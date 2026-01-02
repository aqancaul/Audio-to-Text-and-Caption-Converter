@echo off
REM Script Setup Aplikasi Auto Captioning untuk Windows

REM Cek argumen --compile
set COMPILE_MODE=0
if "%1"=="--compile" set COMPILE_MODE=1

echo Auto Captioning Application - Setup Script
echo ==========================================
if %COMPILE_MODE%==1 (
    echo Mode: COMPILE ^(Portable EXE mode^)
)
echo.

REM Cek apakah Python terinstall
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python tidak terinstall. Silakan install Python 3.8 atau lebih tinggi.
    echo Download dari: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python ditemukan
python --version

REM Cek apakah pip terinstall
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip tidak terinstall. Silakan install pip.
    pause
    exit /b 1
)

echo [OK] pip ditemukan

REM Cek apakah FFmpeg terinstall
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Peringatan: FFmpeg tidak terinstall. Pemrosesan video tidak akan bekerja.
    echo Silakan install FFmpeg:
    echo   - Download dari: https://ffmpeg.org/download.html
    echo   - Atau gunakan chocolatey: choco install ffmpeg
    echo   - Pastikan menambahkan FFmpeg ke PATH Anda
) else (
    echo [OK] FFmpeg ditemukan
    ffmpeg -version | findstr /C:"ffmpeg version"
)

echo.
echo Windows terdeteksi - WASAPI akan digunakan untuk capture audio sistem
echo Catatan: Anda mungkin perlu mengaktifkan "Stereo Mix" di pengaturan suara Windows
echo   (Klik kanan ikon suara ^> Sounds ^> Tab Recording ^> Show Disabled Devices)
echo.

echo Membuat virtual environment Python...
echo.

REM Buat virtual environment jika belum ada
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo Error: Gagal membuat virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment dibuat
) else (
    echo [OK] Virtual environment sudah ada
)

echo.
echo Mengaktifkan virtual environment dan menginstall dependencies...
echo.

REM Aktifkan virtual environment
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo Error: Gagal mengaktifkan virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip terlebih dahulu
python -m pip install --upgrade pip >nul 2>&1

REM Install dependencies Python
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Error: Failed to install dependencies. Please check the error messages above.
    call venv\Scripts\deactivate.bat 2>nul
    pause
    exit /b 1
)

REM Jika mode compile, build portable EXE
if %COMPILE_MODE%==1 (
    echo.
    echo ==========================================
    echo Membangun Portable EXE...
    echo ==========================================
    echo.
    
    REM Install PyInstaller
    echo Menginstall PyInstaller...
    python -m pip install pyinstaller >nul 2>&1
    
    if errorlevel 1 (
        echo Error: Gagal menginstall PyInstaller.
        call venv\Scripts\deactivate.bat 2>nul
        pause
        exit /b 1
    )
    
    REM Buat direktori build
    if exist "build_exe" rmdir /s /q "build_exe"
    if exist "dist_exe" rmdir /s /q "dist_exe"
    mkdir "build_exe" 2>nul
    mkdir "dist_exe" 2>nul
    
    echo Mengompilasi aplikasi dengan PyInstaller...
    echo Ini mungkin memakan waktu beberapa menit...
    echo.
    
    REM Build dengan PyInstaller (single-file executable)
    pyinstaller --name="AutoCaptioning" ^
        --onefile ^
        --windowed ^
        --add-data="requirements.txt;." ^
        --hidden-import=whisper ^
        --hidden-import=faster_whisper ^
        --hidden-import=PyQt6 ^
        --hidden-import=PyQt6.QtCore ^
        --hidden-import=PyQt6.QtGui ^
        --hidden-import=PyQt6.QtWidgets ^
        --hidden-import=sounddevice ^
        --hidden-import=numpy ^
        --hidden-import=torch ^
        --hidden-import=torchaudio ^
        --hidden-import=moviepy ^
        --hidden-import=pycaw ^
        --collect-all=whisper ^
        --collect-all=faster_whisper ^
        --collect-all=torch ^
        --noconfirm ^
        main.py
    
    if errorlevel 1 (
        echo.
        echo Error: PyInstaller build failed.
        call venv\Scripts\deactivate.bat 2>nul
        pause
        exit /b 1
    )
    
    REM Salin executable ke dist_exe
    if exist "dist\AutoCaptioning.exe" (
        copy "dist\AutoCaptioning.exe" "dist_exe\AutoCaptioning.exe" >nul
        echo.
        echo [OK] Portable EXE berhasil dibuat!
        echo   Lokasi: dist_exe\AutoCaptioning.exe
        echo.
        echo Executable bersifat portable dan dapat dijalankan di komputer Windows manapun.
        echo Catatan: Pertama kali menjalankan akan mengunduh model Whisper (mungkin memakan waktu beberapa menit).
        echo.
    ) else (
        echo.
        echo Error: Executable PyInstaller tidak ditemukan di folder dist.
        call venv\Scripts\deactivate.bat 2>nul
        pause
        exit /b 1
    )
    
    REM Bersihkan file build PyInstaller (simpan dist untuk referensi)
    echo Build selesai! Output di: dist_exe\
    echo.
    
) else (
    echo.
    echo ==========================================
    echo IMPORTANT: Virtual environment is active!
    echo ==========================================
    echo.
    echo To run the application:
    echo   1. Activate virtual environment: venv\Scripts\activate
    echo   2. Run application: python main.py
    echo   3. Deactivate when done: deactivate
    echo.
    echo To compile to portable EXE:
    echo   setup.bat --compile
    echo.
    echo Note: The first run will download the Whisper model (may take a few minutes).
    echo Note: You can choose between OpenAI Whisper and Faster Whisper in the application settings.
    echo.
    echo Virtual environment is currently active. You can run 'python main.py' now.
)

pause
