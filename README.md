# Player Re-Identification System

## Overview
This project implements a player re-identification system for sports analytics (Option 2). The system maintains consistent player IDs when players re-enter the frame after going out of view.

## Features
- 🎯 **Enhanced Player Tracking** with Hungarian algorithm
- 🔄 **Kalman Filtering** for motion prediction and smooth trajectories  
- 🧠 **Multi-Modal Features** (appearance + motion + temporal consistency)
- ⚡ **Adaptive Configuration** for different tracking scenarios
- 🎮 **Real-time Processing** with optimized performance
- 📊 **Advanced Analytics** with stability scoring and track quality metrics
- 🔧 **Configurable Parameters** for fine-tuning tracking behavior

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- 4GB+ RAM
- 2GB+ free disk space

### Quick Setup
1. Clone or download this repository
2. Run the automated setup:
```bash
python setup.py
```

### Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required files:
   - **YOLOv11 Model**: [Download](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view) and save as `yolov11_model.pt`
   - **Input Video**: [Download](https://drive.google.com/drive/folders/1Nx6H_n0UUi6L-6i8WknXd4Cv2c3VjZTP) `15sec_input_720p.mp4`

3. Create output directory:
```bash
mkdir -p output/frames
```

### Running the System

**🎯 Quick Start for ID Switching Issues:**
```bash
# For players that come close together (RECOMMENDED)
python3 main.py --tracking_mode crowded
```

**📊 Choose Your Tracking Scenario:**
```bash
# Crowded scenes with players frequently close together
python3 main.py --tracking_mode crowded

# Sparse scenes with few, well-separated players
python3 main.py --tracking_mode sparse

# Fast-moving sports with rapid player movements
python3 main.py --tracking_mode fast

# Scenes with frequent player occlusions
python3 main.py --tracking_mode occlusion

# Default enhanced mode (optimized for general sports)
python3 main.py
```

**🔧 Advanced Usage:**
```bash
python3 main.py --video_path your_video.mp4 --model_path your_model.pt --tracking_mode crowded --output_dir results
```

### Command Line Options
- `--video_path`: Path to input video file (default: `15sec_input_720p.mp4`)
- `--model_path`: Path to YOLOv11 model file (default: `yolov11_model.pt`)
- `--output_dir`: Output directory for results (default: `output`)
- `--tracking_mode`: Tracking optimization mode (default, crowded, sparse, fast, occlusion)
- `--no_frames`: Skip saving individual frames
- `--no_video`: Skip creating output video
- `--quiet`: Suppress progress output

**🎮 Examples:**
```bash
# For ID switching problems when players come close
python3 main.py --tracking_mode crowded --video_path sports_video.mp4

# Fast processing without individual frames
python3 main.py --tracking_mode fast --no_frames --quiet

# Custom model with occlusion-heavy tracking
python3 main.py --model_path custom_model.pt --tracking_mode occlusion
```

### Testing Installation
To verify that everything is set up correctly:
```bash
python3 test_installation.py
```

This will check:
- All required Python packages are installed
- Local modules can be imported
- Required files are present

### Testing Your Custom YOLO Model
After placing your model file, test if it works correctly:
```bash
python3 test_model.py --model_path your_model.pt --video_path your_video.mp4
```

This comprehensive test will:
- ✅ Verify model loading
- ✅ Test detection on sample frames
- ✅ Analyze detection performance on your video
- ✅ Create a sample output image with bounding boxes
- ✅ Provide performance statistics

## Dependencies
- PyTorch
- OpenCV
- Ultralytics (YOLOv11)
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure
```
├── yolov11_model.pt        # YOLO model (PLACE YOUR .pt FILE HERE)
├── 15sec_input_720p.mp4    # Input video (PLACE YOUR VIDEO HERE)
├── main.py                 # Main execution script
├── player_tracker.py       # Core tracking logic
├── feature_extractor.py    # Feature extraction utilities
├── utils.py               # Helper functions
├── test_model.py          # Model testing script
├── setup.py               # Automated setup
├── requirements.txt       # Dependencies
├── README.md             # This file
├── CUSTOM_MODEL_GUIDE.md  # Guide for custom models
└── output/               # Generated results
    ├── frames/           # Individual annotated frames
    ├── tracked_players.mp4  # Output video with tracking
    ├── tracking_summary.md  # Text summary
    └── detection_plot.png   # Performance visualization
```

## Usage
The system processes the input video frame by frame, detects players, and maintains consistent IDs through re-identification when players reappear.

## Author
Computer Vision Assignment - Player Re-Identification