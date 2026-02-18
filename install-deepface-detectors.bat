@echo off
echo ========================================
echo Installing DeepFace Face Detectors
echo ========================================
echo.
echo This will install RetinaFace and MTCNN for better face detection.
echo.

cd python-face-api

echo [1/2] Installing RetinaFace...
pip install retina-face
if %errorlevel% neq 0 (
    echo [WARN] RetinaFace installation failed, will use fallback detectors
)

echo.
echo [2/2] Installing MTCNN...
pip install mtcnn
if %errorlevel% neq 0 (
    echo [WARN] MTCNN installation failed, will use fallback detectors
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
echo The API will now use:
echo   1. RetinaFace (best accuracy)
echo   2. MTCNN (good accuracy)
echo   3. OpenCV (fallback)
echo.
pause
