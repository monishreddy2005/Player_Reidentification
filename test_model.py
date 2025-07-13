#!/usr/bin/env python3
"""
Test script to verify YOLO model is working correctly
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

def test_model_loading(model_path):
    """Test if the YOLO model can be loaded."""
    print(f"🔄 Testing model loading: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("📁 Place your .pt file in the project root directory")
        return None
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        
        # Get model info
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            class_names = model.model.names
            print(f"   Classes detected: {len(class_names)}")
            print(f"   Class names: {list(class_names.values())[:10]}...")  # Show first 10
        
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_single_frame_detection(model, test_image_path=None):
    """Test detection on a single frame."""
    print(f"\n🔄 Testing detection on single frame...")
    
    # Create a test image if none provided
    if test_image_path is None or not os.path.exists(test_image_path):
        print("📝 Creating test image (since no video frame available)")
        # Create a simple test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite("test_frame.jpg", test_img)
        test_image_path = "test_frame.jpg"
    
    try:
        # Load test image
        frame = cv2.imread(test_image_path)
        if frame is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return False
        
        print(f"   Image shape: {frame.shape}")
        
        # Run detection
        results = model(frame, verbose=False)
        print(f"✅ Detection completed!")
        
        # Analyze results
        total_detections = 0
        person_detections = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                total_detections += len(boxes)
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Assuming person class is 0 (COCO standard)
                    if class_id == 0:
                        person_detections += 1
                    
                    print(f"   Detection: Class {class_id}, Confidence: {confidence:.3f}")
        
        print(f"\n📊 Detection Summary:")
        print(f"   Total detections: {total_detections}")
        print(f"   Person detections: {person_detections}")
        
        # Clean up test image
        if test_image_path == "test_frame.jpg":
            os.remove("test_frame.jpg")
        
        return total_detections > 0
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False

def test_video_detection(model, video_path):
    """Test detection on video file."""
    print(f"\n🔄 Testing detection on video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("📁 Place your video file in the project root directory")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    
    # Test on first few frames
    frames_to_test = min(10, total_frames)
    detection_results = []
    
    for frame_idx in range(frames_to_test):
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            results = model(frame, verbose=False)
            
            # Count detections
            frame_detections = 0
            person_detections = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    frame_detections += len(boxes)
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        if class_id == 0:  # Person class
                            person_detections += 1
            
            detection_results.append({
                'frame': frame_idx,
                'total': frame_detections,
                'persons': person_detections
            })
            
            if frame_idx < 3:  # Show details for first 3 frames
                print(f"   Frame {frame_idx}: {frame_detections} total, {person_detections} persons")
        
        except Exception as e:
            print(f"❌ Error processing frame {frame_idx}: {e}")
            cap.release()
            return False
    
    cap.release()
    
    # Summary
    if detection_results:
        avg_detections = np.mean([r['total'] for r in detection_results])
        avg_persons = np.mean([r['persons'] for r in detection_results])
        
        print(f"\n📊 Video Detection Summary:")
        print(f"   Frames tested: {len(detection_results)}")
        print(f"   Average detections per frame: {avg_detections:.1f}")
        print(f"   Average person detections per frame: {avg_persons:.1f}")
        
        if avg_persons > 0:
            print("✅ Model is detecting people in the video!")
            return True
        else:
            print("⚠️  No people detected - check if this is expected for your video")
            return True
    else:
        print("❌ No frames processed successfully")
        return False

def save_sample_output(model, video_path, output_path="sample_output.jpg"):
    """Save a sample frame with detections for visual verification."""
    print(f"\n🔄 Creating sample output: {output_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    # Get a frame from middle of video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame from video")
        return False
    
    try:
        # Run detection
        results = model(frame, verbose=False)
        
        # Draw detections
        output_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Draw bounding box
                    color = (0, 255, 0) if class_id == 0 else (255, 0, 0)  # Green for person
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"Class {class_id}: {confidence:.2f}"
                    cv2.putText(output_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save output
        cv2.imwrite(output_path, output_frame)
        print(f"✅ Sample output saved: {output_path}")
        print(f"   Open this file to visually verify detections")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample output: {e}")
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test YOLO Model")
    parser.add_argument("--model_path", type=str, default="yolov11_model.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--video_path", type=str, default="15sec_input_720p.mp4",
                       help="Path to test video file")
    parser.add_argument("--test_image", type=str, default=None,
                       help="Path to test image (optional)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO Model Testing Tool")
    print("=" * 60)
    
    # Test 1: Model Loading
    model = test_model_loading(args.model_path)
    if model is None:
        print("\n❌ Model loading failed. Cannot proceed with further tests.")
        return False
    
    # Test 2: Single Frame Detection
    detection_works = test_single_frame_detection(model, args.test_image)
    if not detection_works:
        print("\n❌ Single frame detection failed.")
        return False
    
    # Test 3: Video Detection (if video available)
    if os.path.exists(args.video_path):
        video_works = test_video_detection(model, args.video_path)
        if video_works:
            # Test 4: Create Sample Output
            save_sample_output(model, args.video_path)
    else:
        print(f"\n⚠️  Video file not found: {args.video_path}")
        print("   Video testing skipped")
    
    print("\n" + "=" * 60)
    print("🎉 Model testing completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Check 'sample_output.jpg' to visually verify detections")
    print("2. If everything looks good, run: python3 main.py")
    print("3. If detections are poor, consider using a different model")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)