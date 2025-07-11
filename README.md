# Player Re-identification System

A computer vision solution for maintaining consistent player IDs across video frames, designed for sports analytics applications.

## Overview

This project implements a player re-identification system that:
- Detects players in video frames using YOLOv11
- Assigns consistent IDs to players across frames
- Maintains tracking even when players temporarily leave the view
- Re-identifies players when they re-enter the frame

## Features

- **Robust Detection**: Uses fine-tuned YOLOv11 model for accurate player detection
- **Visual Feature Extraction**: Color histograms and shape features for player identification
- **Spatial-Temporal Tracking**: Combines visual similarity with spatial proximity
- **Re-identification**: Maintains player IDs across temporary occlusions
- **Real-time Capable**: Efficient processing suitable for real-time applications

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy, SciPy, scikit-learn

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Player_Reidentification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv11 model:
   - Download from: https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
   - Place the model file as `yolo_model.pt` in the project directory

## Quick Start

### Option 1: Use the Demo Script

```bash
# Create a sample test video and run demo
python demo.py --create-sample

# Or use your own video
python demo.py --video path/to/your/video.mp4
```

### Option 2: Use the Main Script

```bash
# Basic usage
python player_reidentification.py --video input_video.mp4 --output output_with_tracking.mp4

# With custom parameters
python player_reidentification.py \
    --video input_video.mp4 \
    --model custom_model.pt \
    --confidence 0.6 \
    --output tracked_output.mp4 \
    --results tracking_results.json
```

## Usage Examples

### Basic Processing
```python
from player_reidentification import PlayerTracker

# Initialize tracker
tracker = PlayerTracker("yolo_model.pt", confidence_threshold=0.5)

# Process video
results = tracker.process_video("input.mp4", "output.mp4")

# Results contain per-frame tracking data
print(f"Processed {len(results)} frames")
```

### Custom Configuration
```python
tracker = PlayerTracker(
    model_path="yolo_model.pt",
    confidence_threshold=0.6
)

# Adjust tracking parameters
tracker.max_absent_frames = 20  # Frames before ID expires
tracker.similarity_threshold = 0.8  # Stricter matching
```

## Command Line Arguments

- `--video`: Path to input video file (required)
- `--model`: Path to YOLO model file (default: yolo_model.pt)
- `--output`: Path for output video with tracking visualization
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--results`: Path to save tracking results JSON (default: tracking_results.json)

## Output

The system generates:
1. **Annotated Video**: Visual tracking with bounding boxes and player IDs
2. **JSON Results**: Detailed tracking data for each frame
3. **Statistics**: Summary of tracking performance

### JSON Format
```json
[
  {
    "frame": 0,
    "players": [
      {
        "id": 1,
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.85,
        "center": [cx, cy]
      }
    ]
  }
]
```

## Algorithm Details

### Detection
- Uses YOLOv11 fine-tuned for sports player detection
- Applies confidence thresholding to filter detections

### Feature Extraction
- **Visual Features**: RGB color histograms (96 dimensions)
- **Shape Features**: Aspect ratio and bounding box area
- **Combined Features**: 98-dimensional feature vector

### Tracking Algorithm
1. **Detection**: Extract players from current frame
2. **Feature Extraction**: Compute visual and shape features
3. **Similarity Calculation**: Compare with existing players using:
   - Visual similarity (70% weight): Cosine distance between features
   - Spatial similarity (30% weight): Euclidean distance between positions
4. **ID Assignment**: Match to existing player or assign new ID
5. **State Update**: Update player information and remove expired IDs

### Re-identification
- Players absent for up to 30 frames maintain their IDs
- Re-identification based on visual feature similarity
- Spatial proximity used as additional matching criterion

## Evaluation

The system can be evaluated using:
- **Accuracy**: Percentage of correct ID assignments
- **ID Switches**: Number of times a player's ID changes incorrectly
- **Detection Rate**: Percentage of players successfully detected
- **Runtime**: Processing speed (FPS)

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Download the YOLOv11 model from the provided link
   - Ensure the model file is named `yolo_model.pt`

2. **Video Format Issues**
   - Ensure video is in a supported format (MP4, AVI, MOV)
   - Check that OpenCV can read the video file

3. **Memory Issues**
   - Reduce video resolution or process shorter clips
   - Adjust batch processing if needed

4. **Poor Tracking Performance**
   - Adjust confidence threshold (--confidence)
   - Modify similarity threshold in code
   - Increase max_absent_frames for longer occlusions

## Future Improvements

- Deep learning-based re-identification features
- Multi-camera tracking support
- Real-time processing optimizations
- Integration with object tracking algorithms (SORT, DeepSORT)
- Player jersey number recognition
- Team-based color clustering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is developed for educational purposes as part of a computer vision assignment.

## Assignment Context

This implementation addresses **Option 2: Re-Identification in a Single Feed** from the sports analytics assignment, focusing on maintaining consistent player IDs in a 15-second video clip with temporal occlusions and re-entries.