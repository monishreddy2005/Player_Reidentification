# 🎯 Custom YOLO Model Guide

## 📁 Step 1: Place Your Model File

Put your custom YOLO model file (`.pt` extension) in the **root directory**:

```
player-reid-system/
├── your_custom_model.pt    ← Your model here
├── your_video.mp4          ← Your video here
├── main.py
├── test_model.py
└── ...
```

## 🔧 Step 2: Test Your Model

**Always test your model first** to ensure it works correctly:

```bash
# Test with default names
python3 test_model.py

# Or test with custom file names
python3 test_model.py --model_path your_custom_model.pt --video_path your_video.mp4
```

### What the test checks:
- ✅ **Model Loading**: Can the model be loaded without errors?
- ✅ **Detection**: Does the model detect objects in frames?
- ✅ **Person Detection**: Does it detect people/players specifically?
- ✅ **Performance**: How many detections per frame?
- ✅ **Visual Output**: Creates `sample_output.jpg` with bounding boxes

## 🎮 Step 3: Run the Main System

Once testing passes, run the full system:

```bash
# With default file names
python3 main.py

# With custom file names
python3 main.py --model_path your_custom_model.pt --video_path your_video.mp4
```

## 🔍 Expected Model Output

Your YOLO model should detect:
- **Person/Player objects** (typically class ID 0 or classes with "person"/"player" in the name)
- **Bounding boxes** around detected people
- **Confidence scores** (preferably > 0.5 for good tracking)

## ⚠️ Troubleshooting

### Model won't load:
- ✅ Check file path and spelling
- ✅ Ensure `.pt` file is not corrupted
- ✅ Verify it's a valid YOLO model

### No detections:
- ✅ Check if your model detects "person" class (class ID 0)
- ✅ Lower confidence threshold: `--confidence 0.3`
- ✅ Verify video contains people

### Poor tracking:
- ✅ Use higher resolution model
- ✅ Ensure consistent person detections
- ✅ Check lighting/video quality

## 📊 Model Requirements

For best results, your model should:
- **Detect people reliably** (>0.5 confidence)
- **Handle various poses** and orientations
- **Work with your video resolution**
- **Minimize false positives**

## 🎯 Custom Classes

If your model uses custom class names/IDs for players:

1. **Check what classes your model detects:**
   ```bash
   python3 test_model.py --model_path your_model.pt
   ```

2. **Update the class filter in `main.py`** if needed:
   ```python
   # In detect_players() method, modify:
   valid_classes = [0, 1, 2]  # Add your player class IDs
   ```

## 📝 Quick Commands Reference

```bash
# Setup environment
python3 setup.py

# Test installation
python3 test_installation.py

# Test your model
python3 test_model.py --model_path YOUR_MODEL.pt --video_path YOUR_VIDEO.mp4

# Run full system
python3 main.py --model_path YOUR_MODEL.pt --video_path YOUR_VIDEO.mp4
```

## 🎉 Success Indicators

Your model is working well if:
- ✅ Test script shows consistent person detections
- ✅ `sample_output.jpg` shows accurate bounding boxes
- ✅ Average >2-5 person detections per frame (for sports videos)
- ✅ Main system produces tracked output video