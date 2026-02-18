@echo off
echo ========================================
echo Installing Better Face Detection
echo ========================================
echo.

echo This will install face_recognition library for better face detection.
echo.
echo Requirements:
echo - Python 3.8 or higher
echo - Visual C++ Build Tools (Windows)
echo.

pause

echo.
echo [1/3] Installing CMake...
pip install cmake
if %errorlevel% neq 0 (
    echo [FAIL] CMake installation failed
    echo Please install Visual C++ Build Tools from:
    echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dlib...
pip install dlib
if %errorlevel% neq 0 (
    echo [FAIL] dlib installation failed
    echo Trying with pre-built wheel...
    pip install dlib-binary
)

echo.
echo [3/3] Installing face_recognition...
pip install face-recognition
if %errorlevel% neq 0 (
    echo [FAIL] face_recognition installation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Now restart the Python API:
echo   cd python-face-api
echo   python main.py
echo.
echo Then test face detection:
echo   Open test-face-detection.html in browser
echo.
pause
