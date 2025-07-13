#!/usr/bin/env python3
"""
Test script to verify installation and imports
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("ultralytics", "Ultralytics YOLO")
    ]
    
    failed = []
    
    for module, name in imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    return failed

def test_local_modules():
    """Test if local modules can be imported."""
    print("\nTesting local modules...")
    
    modules = [
        "feature_extractor",
        "player_tracker",
        "utils"
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}.py")
        except ImportError as e:
            print(f"❌ {module}.py: {e}")
            failed.append(module)
    
    return failed

def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    
    files = [
        "main.py",
        "requirements.txt",
        "README.md",
        "REPORT.md"
    ]
    
    missing = []
    
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing.append(file)
    
    return missing

def main():
    """Main test function."""
    print("=" * 50)
    print("Player Re-ID System - Installation Test")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test local modules
    failed_modules = test_local_modules()
    
    # Check files
    missing_files = check_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if not failed_imports and not failed_modules and not missing_files:
        print("🎉 All tests passed! System ready to use.")
        print("\nNext steps:")
        print("1. Download the YOLOv11 model and video files")
        print("2. Run: python main.py")
        return True
    else:
        print("⚠️  Some issues found:")
        
        if failed_imports:
            print(f"   Failed imports: {', '.join(failed_imports)}")
            print("   Run: pip install -r requirements.txt")
        
        if failed_modules:
            print(f"   Missing modules: {', '.join(failed_modules)}")
        
        if missing_files:
            print(f"   Missing files: {', '.join(missing_files)}")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)