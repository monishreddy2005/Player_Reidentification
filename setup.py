#!/usr/bin/env python3
"""
Setup script for Player Re-Identification System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_files():
    """Check if required files exist."""
    print("Checking for required files...")
    
    files_needed = [
        ("YOLOv11 Model", "yolov11_model.pt"),
        ("Input Video", "15sec_input_720p.mp4")
    ]
    
    missing_files = []
    for name, filename in files_needed:
        if os.path.exists(filename):
            print(f"✅ {name}: {filename}")
        else:
            print(f"❌ {name}: {filename} (MISSING)")
            missing_files.append((name, filename))
    
    if missing_files:
        print("\n📁 Please download the following files:")
        print("Model: https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view")
        print("Videos: https://drive.google.com/drive/folders/1Nx6H_n0UUi6L-6i8WknXd4Cv2c3VjZTP")
        print("\nPlace them in the project directory and run setup again.")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating output directories...")
    directories = ["output", "output/frames"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Player Re-Identification System - Setup")
    print("=" * 60)
    
    # Install dependencies
    if not install_requirements():
        print("Setup failed. Please check error messages above.")
        return
    
    # Create directories
    create_directories()
    
    # Check for required files
    if check_files():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test your model: python3 test_model.py")
        print("2. Run the system: python3 main.py")
    else:
        print("\n⚠️  Setup completed but some files are missing.")
        print("Please download the required files and run setup again.")
        print("\nOnce files are ready:")
        print("1. Test your model: python3 test_model.py")
        print("2. Run the system: python3 main.py")

if __name__ == "__main__":
    main()