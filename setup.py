#!/usr/bin/env python3
"""
Setup script for Player Re-identification System

This script handles the installation and setup of the player re-identification system.
"""

import os
import sys
import subprocess
import argparse


def run_command(command, check=True):
    """Run a shell command and handle errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Update pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip"):
        print("Warning: Failed to upgrade pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt")
        if success:
            print("✓ Dependencies installed successfully")
            return True
        else:
            print("✗ Failed to install dependencies")
            return False
    else:
        print("✗ requirements.txt not found")
        return False


def setup_directory_structure():
    """Create necessary directories."""
    print("\nSetting up directory structure...")
    
    directories = [
        "models",
        "videos", 
        "output",
        "results",
        "evaluation_output"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")


def check_model_availability():
    """Check if YOLO model is available."""
    print("\nChecking model availability...")
    
    model_paths = ["yolo_model.pt", "models/yolo_model.pt", "yolov8n.pt"]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"✓ Model found: {model_path}")
            return True
    
    print("⚠ No YOLO model found. You need to:")
    print("  1. Download the fine-tuned model from:")
    print("     https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view")
    print("  2. Place it as 'yolo_model.pt' in the project directory")
    print("  3. Or use the demo script which will download a standard YOLO model")
    
    return False


def run_basic_test():
    """Run a basic test to verify installation."""
    print("\nRunning basic test...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import torch
        
        print("✓ All required packages imported successfully")
        
        # Test OpenCV
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test PyTorch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def create_sample_scripts():
    """Create sample usage scripts."""
    print("\nCreating sample scripts...")
    
    # Create a quick test script
    test_script = '''#!/usr/bin/env python3
"""
Quick test script for player re-identification system
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from demo import main as demo_main

if __name__ == "__main__":
    print("Running quick test with sample video...")
    sys.argv = ["quick_test.py", "--create-sample"]
    demo_main()
'''
    
    with open("quick_test.py", "w") as f:
        f.write(test_script)
    
    # Make it executable
    if os.name != 'nt':  # Not Windows
        os.chmod("quick_test.py", 0o755)
    
    print("✓ Created quick_test.py")


def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print()
    print("Quick Start Options:")
    print()
    print("1. Run with sample video:")
    print("   python demo.py --create-sample")
    print()
    print("2. Run with your own video:")
    print("   python demo.py --video path/to/your/video.mp4")
    print()
    print("3. Use the main script:")
    print("   python player_reidentification.py --video input.mp4 --output output.mp4")
    print()
    print("4. Run quick test:")
    print("   python quick_test.py")
    print()
    print("5. Evaluate results:")
    print("   python evaluate.py --results tracking_results.json --plots")
    print()
    print("Important Notes:")
    print("- For best results, download the fine-tuned YOLO model")
    print("- Place videos in the 'videos/' directory")
    print("- Output files will be saved in 'output/' and 'results/' directories")
    print("- Check README.md for detailed documentation")
    print()
    print("If you encounter issues, check the troubleshooting section in README.md")


def main():
    parser = argparse.ArgumentParser(description="Setup Player Re-identification System")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run tests, skip installation")
    
    args = parser.parse_args()
    
    print("Player Re-identification System Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    if not args.test_only:
        # Install dependencies
        if not args.skip_deps:
            if not install_dependencies():
                print("\nSetup failed during dependency installation")
                return 1
        
        # Setup directories
        setup_directory_structure()
        
        # Create sample scripts
        create_sample_scripts()
    
    # Check model availability
    check_model_availability()
    
    # Run basic test
    if not run_basic_test():
        print("\nSetup completed with warnings - some tests failed")
        print("The system may still work, but you might encounter issues")
    
    # Print usage instructions
    if not args.test_only:
        print_usage_instructions()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())