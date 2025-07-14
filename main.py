#!/usr/bin/env python3
"""
Player Re-Identification System
Main script for processing video and tracking players across frames.
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not installed. Please run: pip install ultralytics")
    sys.exit(1)

from robust_tracker import HybridTracker, RobustPlayerTracker
from utils import (
    download_model_if_needed, create_color_palette, draw_tracks_on_frame,
    save_frame_with_tracks, create_tracking_summary, create_detection_plot,
    validate_video_file, get_video_info, filter_player_detections,
    create_output_video, log_processing_stats, ensure_output_dir
)

class PlayerReIDSystem:
    """Main class for player re-identification system."""
    
    def __init__(self, model_path: str, output_dir: str = "output", 
                 tracking_mode: str = "robust"):
        """Initialize the re-identification system."""
        self.model_path = model_path
        self.output_dir = output_dir
        self.model = None
        self.tracking_mode = tracking_mode
        
        # Initialize robust tracker
        print(f"🎯 Initializing tracking system in {tracking_mode} mode...")
        
        if tracking_mode.lower() in ["robust", "stable", "conservative"]:
            self.tracker = RobustPlayerTracker(debug_mode=True)
        elif tracking_mode.lower() in ["basic", "simple", "original"]:
            self.tracker = HybridTracker(mode="basic")
        else:
            # Default to robust for best stability
            self.tracker = RobustPlayerTracker(debug_mode=True)
            print("⚠️ Unknown mode, defaulting to robust tracking")
        
        self.colors = create_color_palette(20)  # Support up to 20 different players
        
        # Ensure output directory exists
        ensure_output_dir(self.output_dir)
        
    def load_model(self):
        """Load the YOLOv11 model."""
        print("Loading YOLOv11 model...")
        try:
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def detect_players(self, frame: np.ndarray, 
                      confidence_threshold: float = 0.5) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect players in a frame using YOLOv11."""
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name if available
                    class_name = "unknown"
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                        class_name = self.model.model.names.get(class_id, f"class_{class_id}")
                    
                    # Filter for person/player classes
                    # Common person class IDs: 0 (COCO person), or custom player classes
                    valid_classes = [0]  # Add more class IDs if your model uses different ones
                    
                    # Also accept if class name contains 'person' or 'player'
                    if (class_id in valid_classes or 
                        'person' in class_name.lower() or 
                        'player' in class_name.lower()) and confidence >= confidence_threshold:
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        detections.append((bbox, float(confidence)))
        
        return detections
    
    def process_video(self, video_path: str, save_frames: bool = True, 
                     save_video: bool = True, show_progress: bool = True):
        """Process video for player re-identification."""
        print(f"Processing video: {video_path}")
        
        # Validate video file
        if not validate_video_file(video_path):
            return False
        
        # Get video information
        video_info = get_video_info(video_path)
        print(f"Video info: {video_info['total_frames']} frames, "
              f"{video_info['fps']:.1f} FPS, "
              f"{video_info['width']}x{video_info['height']}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Initialize tracking
        self.tracker.reset()
        processed_frames = []
        frame_count = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect players in current frame
            detections = self.detect_players(frame)
            
            # Filter detections
            filtered_detections = filter_player_detections(
                detections, min_confidence=0.5, min_area=1000)
            
            # Update tracker
            tracks = self.tracker.update(filtered_detections, frame)
            
            # Draw tracks on frame
            frame_with_tracks = draw_tracks_on_frame(frame, tracks, self.colors)
            
            # Log progress
            if show_progress and frame_count % 10 == 0:
                log_processing_stats(frame_count, video_info['total_frames'], 
                                   len(filtered_detections), len(tracks))
            
            # Save frame if requested
            if save_frames:
                save_frame_with_tracks(frame_with_tracks, frame_count, 
                                     f"{self.output_dir}/frames")
            
            # Store frame for video creation
            if save_video:
                processed_frames.append(frame_with_tracks)
        
        cap.release()
        
        # Create output video
        if save_video and processed_frames:
            output_video_path = f"{self.output_dir}/tracked_players.mp4"
            create_output_video(processed_frames, output_video_path, video_info['fps'])
        
        # Generate tracking summary
        track_history = self.tracker.get_track_history()
        summary_path = f"{self.output_dir}/tracking_summary.md"
        create_tracking_summary(track_history, summary_path)
        
        # Create detection plot
        plot_path = f"{self.output_dir}/detection_plot.png"
        create_detection_plot(track_history, plot_path)
        
        # Print final statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🎉 Processing completed!")
        print(f"📊 Final Statistics:")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total unique players detected: {len(track_history)}")
        print(f"   Processing time: {processing_time:.1f} seconds")
        print(f"   Average FPS: {frame_count / processing_time:.1f}")
        print(f"   Results saved to: {self.output_dir}/")
        
        # Show tracking statistics if available
        if hasattr(self.tracker, 'get_stats'):
            stats = self.tracker.get_stats()
            print(f"\n🔍 Tracking Statistics:")
            print(f"   Tracking mode: {self.tracking_mode}")
            print(f"   Total tracks created: {stats.get('total_tracks_created', 'N/A')}")
            print(f"   ID switch rate: {stats.get('id_switch_rate', 0):.1f}%")
            print(f"   Total assignments: {stats.get('total_assignments', 'N/A')}")
        
        # Show stability scores
        if track_history:
            print(f"\n📈 Track Stability Scores:")
            for track_id, track_info in track_history.items():
                stability = track_info.get('stability_score', 0)
                detections = track_info.get('total_detections', 0)
                print(f"   Player {track_id}: {stability:.2f} stability, {detections} detections")
        
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Player Re-Identification System")
    parser.add_argument("--video_path", type=str, default="15sec_input_720p.mp4",
                       help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="yolov11_model.pt",
                       help="Path to YOLOv11 model file")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument("--no_frames", action="store_true",
                       help="Don't save individual frames")
    parser.add_argument("--no_video", action="store_true",
                       help="Don't create output video")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress output")
    parser.add_argument("--tracking_mode", type=str, default="robust",
                       choices=["robust", "stable", "conservative", "basic", "simple", "original"],
                       help="Tracking mode: robust (recommended), basic (simple), stable, conservative")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Player Re-Identification System")
    print("=" * 60)
    
    # Check if model file exists
    model_url = "https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view"
    if not download_model_if_needed(model_url, args.model_path):
        print("\nPlease download the required files:")
        print("1. YOLOv11 model: Place as 'yolov11_model.pt'")
        print("2. Video file: Place as '15sec_input_720p.mp4'")
        print("\nDownload links:")
        print(f"- Model: {model_url}")
        print("- Videos: https://drive.google.com/drive/folders/1Nx6H_n0UUi6L-6i8WknXd4Cv2c3VjZTP")
        return
    
    # Initialize system
    reid_system = PlayerReIDSystem(args.model_path, args.output_dir, args.tracking_mode)
    
    # Load model
    if not reid_system.load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Process video
    success = reid_system.process_video(
        args.video_path,
        save_frames=not args.no_frames,
        save_video=not args.no_video,
        show_progress=not args.quiet
    )
    
    if success:
        print("\n✅ Processing completed successfully!")
        print(f"Check the '{args.output_dir}' directory for results.")
    else:
        print("\n❌ Processing failed.")

if __name__ == "__main__":
    main()