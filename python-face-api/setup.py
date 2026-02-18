#!/usr/bin/env python3
"""
Setup script for "AURA"-Automated Unified Recognition for Attendance Face Recognition API
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up \"AURA\"-Automated Unified Recognition for Attendance Face Recognition API...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    
    # Install requirements
    print("\n📦 Installing Python packages...")
    if not run_command("pip install -r requirements.txt"):
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Download YOLOv8 face model (optional)
    print("\n🤖 Downloading YOLOv8 face detection model...")
    yolo_commands = [
        "pip install ultralytics",
        "python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"",  # Download base model
    ]
    
    for cmd in yolo_commands:
        run_command(cmd)
    
    print("\n✅ Setup completed!")
    print("\n🚀 To start the API server, run:")
    print("   python main.py")
    print("\n🌐 API will be available at: http://localhost:8000")
    print("📚 API docs will be at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()