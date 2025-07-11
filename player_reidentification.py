import cv2
import numpy as np
from ultralytics import YOLO
import torch
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import DBSCAN
import os
import argparse
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm


class PlayerTracker:
    """
    Player re-identification system that maintains consistent IDs across frames
    even when players temporarily leave the view.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the player tracker.
        
        Args:
            model_path: Path to the YOLOv11 model
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Tracking state
        self.players = {}  # player_id: PlayerInfo
        self.next_id = 1
        self.max_absent_frames = 30  # Max frames a player can be absent before ID expires
        self.similarity_threshold = 0.7  # Threshold for feature similarity
        
    def extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract visual features from a player bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Feature vector
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(128)  # Return zero vector for invalid bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(128)
        
        # Resize to standard size
        player_region = cv2.resize(player_region, (64, 128))
        
        # Extract color histogram features
        hist_b = cv2.calcHist([player_region], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([player_region], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([player_region], [2], None, [32], [0, 256])
        
        # Flatten and normalize
        features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # Add shape features
        aspect_ratio = (x2 - x1) / max(1, (y2 - y1))
        area = (x2 - x1) * (y2 - y1)
        shape_features = np.array([aspect_ratio, area / 10000])  # Normalize area
        
        return np.concatenate([features, shape_features])
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                           pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate similarity between two player instances.
        
        Args:
            features1, features2: Visual feature vectors
            pos1, pos2: Center positions (x, y)
            
        Returns:
            Similarity score (0-1)
        """
        # Visual similarity
        if len(features1) == 0 or len(features2) == 0:
            visual_sim = 0.0
        else:
            visual_sim = 1 - cosine(features1, features2)
        
        # Spatial similarity (closer positions are more similar)
        spatial_dist = euclidean(pos1, pos2)
        spatial_sim = 1 / (1 + spatial_dist / 100)  # Normalize by image size assumption
        
        # Combined similarity
        return 0.7 * visual_sim + 0.3 * spatial_sim
    
    def update_tracking(self, frame: np.ndarray, detections: List) -> List[Dict]:
        """
        Update player tracking with new detections.
        
        Args:
            frame: Current frame
            detections: List of YOLO detections
            
        Returns:
            List of tracked players with IDs
        """
        current_players = []
        
        # Extract features for all detections
        detection_data = []
        for det in detections:
            bbox = det.boxes.xyxy[0].cpu().numpy()
            conf = det.boxes.conf[0].cpu().numpy()
            
            if conf >= self.confidence_threshold:
                features = self.extract_features(frame, bbox)
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                detection_data.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'features': features,
                    'center': (center_x, center_y)
                })
        
        # Match detections to existing players
        assigned_ids = set()
        for det_data in detection_data:
            best_match_id = None
            best_similarity = 0
            
            for player_id, player_info in self.players.items():
                if player_id in assigned_ids:
                    continue
                    
                similarity = self.calculate_similarity(
                    det_data['features'],
                    player_info['features'],
                    det_data['center'],
                    player_info['last_position']
                )
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = player_id
            
            # Assign ID
            if best_match_id is not None:
                player_id = best_match_id
                assigned_ids.add(player_id)
            else:
                player_id = self.next_id
                self.next_id += 1
            
            # Update player info
            self.players[player_id] = {
                'features': det_data['features'],
                'last_position': det_data['center'],
                'bbox': det_data['bbox'],
                'confidence': det_data['confidence'],
                'last_seen_frame': self.current_frame,
                'absent_frames': 0
            }
            
            current_players.append({
                'id': player_id,
                'bbox': det_data['bbox'].tolist(),
                'confidence': float(det_data['confidence']),
                'center': det_data['center']
            })
        
        # Update absent frames and remove old players
        players_to_remove = []
        for player_id, player_info in self.players.items():
            if player_id not in assigned_ids:
                player_info['absent_frames'] += 1
                if player_info['absent_frames'] > self.max_absent_frames:
                    players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            del self.players[player_id]
        
        return current_players
    
    def process_video(self, video_path: str, output_path: str = None) -> List[List[Dict]]:
        """
        Process entire video and track players.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            
        Returns:
            List of tracking results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_tracking_results = []
        self.current_frame = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, verbose=False)
            detections = [r for r in results if r.boxes is not None and len(r.boxes) > 0]
            
            # Update tracking
            if detections:
                tracked_players = self.update_tracking(frame, detections)
            else:
                tracked_players = []
            
            all_tracking_results.append(tracked_players)
            
            # Draw tracking results
            if output_path:
                annotated_frame = self.draw_tracking(frame.copy(), tracked_players)
                out.write(annotated_frame)
            
            self.current_frame += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if output_path:
            out.release()
        
        return all_tracking_results
    
    def draw_tracking(self, frame: np.ndarray, tracked_players: List[Dict]) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            tracked_players: List of tracked player data
            
        Returns:
            Annotated frame
        """
        for player in tracked_players:
            player_id = player['id']
            bbox = player['bbox']
            confidence = player['confidence']
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Generate color based on ID
            color = ((player_id * 50) % 255, (player_id * 100) % 255, (player_id * 150) % 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and confidence
            label = f"Player {player_id}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


def download_model(model_url: str, model_path: str):
    """Download the YOLOv11 model if not present."""
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}")
        try:
            import gdown
            gdown.download(model_url, model_path, quiet=False)
        except ImportError:
            print("Please install gdown: pip install gdown")
            print(f"Or manually download the model from: {model_url}")
            print(f"And place it at: {model_path}")
            raise
    else:
        print(f"Model already exists at {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Player Re-identification System")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="yolo_model.pt", help="Path to YOLO model")
    parser.add_argument("--output", help="Path for output video")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold for detections")
    parser.add_argument("--results", default="tracking_results.json", 
                       help="Path to save tracking results")
    
    args = parser.parse_args()
    
    # Download model if needed
    model_url = "https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view"
    if not os.path.exists(args.model):
        print("Model not found. Please download it manually from:")
        print(model_url)
        print(f"And place it at: {args.model}")
        return
    
    # Initialize tracker
    tracker = PlayerTracker(args.model, args.confidence)
    
    # Process video
    print(f"Processing video: {args.video}")
    tracking_results = tracker.process_video(args.video, args.output)
    
    # Save results
    with open(args.results, 'w') as f:
        json.dump(tracking_results, f, indent=2)
    
    print(f"Tracking results saved to: {args.results}")
    if args.output:
        print(f"Annotated video saved to: {args.output}")
    
    # Print statistics
    total_frames = len(tracking_results)
    total_detections = sum(len(frame_results) for frame_results in tracking_results)
    unique_players = len(tracker.players)
    
    print(f"\nStatistics:")
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Unique players tracked: {unique_players}")


if __name__ == "__main__":
    main()