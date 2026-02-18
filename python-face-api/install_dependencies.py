#!/usr/bin/env python
"""
"AURA"-Automated Unified Recognition for Attendance Face Recognition - Complete Dependency Installation
Ensures all required packages are installed correctly
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     "AURA"-Automated Unified Recognition for Attendance - Complete Setup                      ║
║     Installing all dependencies for ArcFace-powered recognition      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    packages = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install fastapi==0.104.1", "FastAPI (API framework)"),
        ("pip install uvicorn[standard]==0.24.0", "Uvicorn (ASGI server)"),
        ("pip install numpy==1.24.3", "NumPy (numerical computing)"),
        ("pip install opencv-python==4.8.1.78", "OpenCV (computer vision)"),
        ("pip install Pillow==10.0.1", "Pillow (image processing)"),
        ("pip install scipy==1.11.3", "SciPy (scientific computing)"),
        ("pip install tensorflow==2.20.0", "TensorFlow (deep learning)"),
        ("pip install tf-keras==2.15.0", "Keras (neural networks)"),
        ("pip install deepface==0.0.98", "DeepFace (face recognition)"),
        ("pip install retina-face==0.0.13", "RetinaFace (face detection)"),
        ("pip install mtcnn==0.1.1", "MTCNN (face detection)"),
        ("pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu", "PyTorch (for better detection)"),
    ]

    print("\n📦 Installing packages...\n")
    
    successful = []
    failed = []

    for cmd, desc in packages:
        if run_command(cmd, desc):
            successful.append(desc)
        else:
            failed.append(desc)

    print(f"\n{'='*70}")
    print("📊 INSTALLATION SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful: {len(successful)}")
    for item in successful:
        print(f"   ✓ {item}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for item in failed:
            print(f"   ✗ {item}")
    
    print(f"\n{'='*70}")
    print("🎉 Installation Complete!")
    print(f"{'='*70}")
    
    if failed:
        print("\n⚠️  Some packages failed to install. You may experience limitations.")
        print("   The system will still work with available packages.\n")
    else:
        print("\n✅ All packages installed successfully!")
        print("   Your face recognition system is ready!\n")
    
    print("Next step: Start the Python API")
    print("   python main.py\n")

if __name__ == "__main__":
    main()
