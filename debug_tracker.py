#!/usr/bin/env python3
"""
Debug script for tracking issues - helps identify and fix problems
"""

import cv2
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

from robust_tracker import RobustPlayerTracker, HybridTracker
from player_tracker import PlayerTracker
from feature_extractor import FeatureExtractor
from utils import create_color_palette, validate_video_file, get_video_info

class TrackingDebugger:
    """Debug tracking issues and compare different approaches."""
    
    def __init__(self, model_path: str, video_path: str):
        """Initialize debugger."""
        self.model_path = model_path
        self.video_path = video_path
        self.model = None
        self.feature_extractor = FeatureExtractor()
        
        # Initialize multiple trackers for comparison
        self.trackers = {
            'robust': RobustPlayerTracker(debug_mode=False),
            'basic': PlayerTracker(similarity_threshold=0.6),
        }
        
        self.colors = create_color_palette(20)
        
    def load_model(self):
        """Load YOLO model."""
        print(f"🔄 Loading model: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_players(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect players in frame."""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for person class
                    if class_id == 0 and confidence >= 0.5:
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        detections.append((bbox, float(confidence)))
        
        return detections
    
    def debug_feature_extraction(self, frame: np.ndarray, detections: List):
        """Debug feature extraction process."""
        print(f"\n🔍 Debugging feature extraction for {len(detections)} detections:")
        
        for i, (bbox, conf) in enumerate(detections):
            print(f"   Detection {i+1}: bbox={bbox}, conf={conf:.3f}")
            
            try:
                # Test appearance features
                app_features = self.feature_extractor.extract_appearance_features(frame, bbox)
                pos_features = self.feature_extractor.extract_positional_features(bbox, frame.shape[:2])
                
                print(f"      Appearance features: shape={app_features.shape}, "
                      f"mean={np.mean(app_features):.3f}, std={np.std(app_features):.3f}")
                print(f"      Position features: shape={pos_features.shape}, "
                      f"values={pos_features[:5]}...")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(app_features)) or np.any(np.isinf(app_features)):
                    print(f"      ⚠️ WARNING: Invalid appearance features detected!")
                if np.any(np.isnan(pos_features)) or np.any(np.isinf(pos_features)):
                    print(f"      ⚠️ WARNING: Invalid position features detected!")
                    
            except Exception as e:
                print(f"      ❌ Feature extraction failed: {e}")
    
    def debug_similarity_computation(self, frame: np.ndarray, detections: List):
        """Debug similarity computation between detections."""
        print(f"\n🔍 Debugging similarity computation:")
        
        if len(detections) < 2:
            print("   Need at least 2 detections for similarity testing")
            return
        
        # Extract features for first two detections
        try:
            bbox1, conf1 = detections[0]
            bbox2, conf2 = detections[1]
            
            features1 = self.feature_extractor.extract_appearance_features(frame, bbox1)
            features2 = self.feature_extractor.extract_appearance_features(frame, bbox2)
            
            similarity = self.feature_extractor.compute_similarity(features1, features2)
            
            print(f"   Detection 1 vs 2: similarity = {similarity:.3f}")
            
            # Test distance
            center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
            center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            print(f"   Distance between centers: {distance:.1f} pixels")
            
        except Exception as e:
            print(f"   ❌ Similarity computation failed: {e}")
    
    def compare_trackers(self, max_frames: int = 100):
        """Compare different tracking approaches."""
        print(f"\n🔍 Comparing tracking approaches on {max_frames} frames:")
        
        if not validate_video_file(self.video_path):
            return False
        
        cap = cv2.VideoCapture(self.video_path)
        video_info = get_video_info(self.video_path)
        
        frame_count = 0
        tracker_results = {name: [] for name in self.trackers.keys()}
        tracker_stats = {name: {'id_switches': 0, 'total_tracks': 0} for name in self.trackers.keys()}
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get detections
            detections = self.detect_players(frame)
            
            if frame_count == 1:
                print(f"   First frame: {len(detections)} detections")
                self.debug_feature_extraction(frame, detections[:2])
                self.debug_similarity_computation(frame, detections)
            
            # Update all trackers
            for tracker_name, tracker in self.trackers.items():
                try:
                    tracks = tracker.update(detections, frame)
                    tracker_results[tracker_name].append(len(tracks))
                    
                    # Count unique IDs
                    current_ids = set(track_id for track_id, _, _ in tracks)
                    if hasattr(tracker, 'get_stats'):
                        stats = tracker.get_stats()
                        tracker_stats[tracker_name] = stats
                    else:
                        if frame_count == 1:
                            tracker_stats[tracker_name]['total_tracks'] = len(current_ids)
                        else:
                            tracker_stats[tracker_name]['total_tracks'] = max(
                                tracker_stats[tracker_name]['total_tracks'], 
                                max(current_ids) if current_ids else 0
                            )
                    
                except Exception as e:
                    print(f"   ❌ {tracker_name} tracker failed on frame {frame_count}: {e}")
                    tracker_results[tracker_name].append(0)
            
            if frame_count % 20 == 0:
                print(f"   Frame {frame_count}: ", end="")
                for name in self.trackers.keys():
                    tracks_count = tracker_results[name][-1] if tracker_results[name] else 0
                    print(f"{name}={tracks_count} ", end="")
                print()
        
        cap.release()
        
        # Print comparison results
        print(f"\n📊 Tracking Comparison Results ({frame_count} frames):")
        for tracker_name, results in tracker_results.items():
            if results:
                avg_tracks = np.mean(results)
                std_tracks = np.std(results)
                stats = tracker_stats[tracker_name]
                
                print(f"   {tracker_name.upper()} Tracker:")
                print(f"      Average active tracks: {avg_tracks:.1f} ± {std_tracks:.1f}")
                print(f"      Total unique IDs: {stats.get('total_tracks_created', stats.get('total_tracks', 'N/A'))}")
                print(f"      ID switch rate: {stats.get('id_switch_rate', 'N/A')}")
        
        return True
    
    def analyze_detection_quality(self, num_frames: int = 50):
        """Analyze detection quality and consistency."""
        print(f"\n🔍 Analyzing detection quality over {num_frames} frames:")
        
        if not validate_video_file(self.video_path):
            return False
        
        cap = cv2.VideoCapture(self.video_path)
        
        detection_counts = []
        confidence_scores = []
        bbox_sizes = []
        
        for frame_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect_players(frame)
            detection_counts.append(len(detections))
            
            for bbox, conf in detections:
                confidence_scores.append(conf)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                bbox_sizes.append(w * h)
        
        cap.release()
        
        if detection_counts:
            print(f"   Detection count: mean={np.mean(detection_counts):.1f}, "
                  f"std={np.std(detection_counts):.1f}, "
                  f"range={min(detection_counts)}-{max(detection_counts)}")
        
        if confidence_scores:
            print(f"   Confidence scores: mean={np.mean(confidence_scores):.3f}, "
                  f"std={np.std(confidence_scores):.3f}")
        
        if bbox_sizes:
            print(f"   Bbox sizes: mean={np.mean(bbox_sizes):.0f}, "
                  f"std={np.std(bbox_sizes):.0f}")
        
        return True
    
    def create_debug_video(self, output_path: str = "debug_tracking.mp4", max_frames: int = 150):
        """Create debug video showing tracking results."""
        print(f"\n🎥 Creating debug video: {output_path}")
        
        if not validate_video_file(self.video_path):
            return False
        
        cap = cv2.VideoCapture(self.video_path)
        video_info = get_video_info(self.video_path)
        
        # Initialize tracker
        tracker = RobustPlayerTracker(debug_mode=False)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, min(video_info['fps'], 15), 
                             (video_info['width'], video_info['height']))
        
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get detections and tracks
            detections = self.detect_players(frame)
            tracks = tracker.update(detections, frame)
            
            # Draw debug info
            debug_frame = frame.copy()
            
            # Draw raw detections in red
            for bbox, conf in detections:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(debug_frame, f"Det: {conf:.2f}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw tracks in green
            for track_id, bbox, conf in tracks:
                x1, y1, x2, y2 = bbox
                color = self.colors[track_id % len(self.colors)]
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_frame, f"ID: {track_id}", (x1, y1-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame info
            cv2.putText(debug_frame, f"Frame: {frame_count}, Dets: {len(detections)}, Tracks: {len(tracks)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(debug_frame)
        
        cap.release()
        out.release()
        
        print(f"✅ Debug video saved: {output_path}")
        return True

def main():
    """Main debug function."""
    parser = argparse.ArgumentParser(description="Debug Player Tracking Issues")
    parser.add_argument("--model_path", type=str, default="yolov11_model.pt",
                       help="Path to YOLO model")
    parser.add_argument("--video_path", type=str, default="15sec_input_720p.mp4",
                       help="Path to video file")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Maximum frames to process")
    parser.add_argument("--create_video", action="store_true",
                       help="Create debug video")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🐛 Player Tracking Debugger")
    print("=" * 60)
    
    # Initialize debugger
    debugger = TrackingDebugger(args.model_path, args.video_path)
    
    # Load model
    if not debugger.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Run debug analyses
    print("\n" + "="*60)
    print("🔍 DEBUGGING ANALYSES")
    print("="*60)
    
    # 1. Analyze detection quality
    debugger.analyze_detection_quality(num_frames=50)
    
    # 2. Compare tracking approaches
    debugger.compare_trackers(max_frames=args.max_frames)
    
    # 3. Create debug video if requested
    if args.create_video:
        debugger.create_debug_video(max_frames=args.max_frames)
    
    print(f"\n" + "="*60)
    print("🎯 RECOMMENDATIONS")
    print("="*60)
    print("1. Use 'robust' mode for best ID stability")
    print("2. Check detection quality - need consistent person detections")
    print("3. If IDs still switch, try lowering similarity threshold")
    print("4. For very close players, increase appearance weight")
    print("\n🚀 Run with: python3 main.py --tracking_mode robust")

if __name__ == "__main__":
    main()