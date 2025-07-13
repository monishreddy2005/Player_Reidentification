import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict
import os
from pathlib import Path

def download_model_if_needed(model_url: str, model_path: str) -> bool:
    """
    Download the YOLOv11 model if it doesn't exist.
    Note: This is a placeholder - in a real scenario, you'd implement actual download logic.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print(f"Please download the model from: {model_url}")
        print("Place it in the current directory as 'yolov11_model.pt'")
        return False
    return True

def create_color_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    """Create a color palette for visualizing different player IDs."""
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors

def draw_tracks_on_frame(frame: np.ndarray, 
                        tracks: List[Tuple[int, Tuple[int, int, int, int], float]], 
                        colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Draw bounding boxes and IDs on frame."""
    frame_with_tracks = frame.copy()
    
    for track_id, bbox, confidence in tracks:
        x1, y1, x2, y2 = bbox
        
        # Get color for this track ID
        color = colors[track_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(frame_with_tracks, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and confidence
        label = f"ID: {track_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(frame_with_tracks, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(frame_with_tracks, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame_with_tracks

def save_frame_with_tracks(frame: np.ndarray, frame_num: int, output_dir: str):
    """Save frame with tracks to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"frame_{frame_num:06d}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)

def create_tracking_summary(track_history: Dict[int, Dict], output_path: str):
    """Create a summary of tracking results."""
    summary = []
    summary.append("# Player Re-Identification Tracking Summary\n")
    
    total_tracks = len(track_history)
    summary.append(f"**Total Unique Players Detected:** {total_tracks}\n")
    
    if total_tracks > 0:
        avg_detections = np.mean([track['total_detections'] for track in track_history.values()])
        summary.append(f"**Average Detections per Player:** {avg_detections:.1f}\n")
        
        summary.append("## Individual Track Details\n")
        for track_id, track_info in sorted(track_history.items()):
            summary.append(f"- **Player {track_id}:**")
            summary.append(f"  - Total Detections: {track_info['total_detections']}")
            summary.append(f"  - Last Seen Frame: {track_info['last_seen_frame']}")
            summary.append(f"  - Final Confidence: {track_info['confidence']:.3f}")
            summary.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))

def create_detection_plot(track_history: Dict[int, Dict], output_path: str):
    """Create a plot showing detection timeline for each player."""
    if not track_history:
        return
        
    plt.figure(figsize=(12, 8))
    
    track_ids = list(track_history.keys())
    y_positions = range(len(track_ids))
    
    for i, track_id in enumerate(track_ids):
        track = track_history[track_id]
        plt.barh(i, track['total_detections'], 
                label=f"Player {track_id}", alpha=0.7)
    
    plt.xlabel('Total Detections')
    plt.ylabel('Player ID')
    plt.title('Player Detection Frequency')
    plt.yticks(y_positions, [f"Player {tid}" for tid in track_ids])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def validate_video_file(video_path: str) -> bool:
    """Validate that video file exists and can be opened."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False
    
    cap.release()
    return True

def get_video_info(video_path: str) -> Dict:
    """Get basic information about the video."""
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['total_frames'] / info['fps']
    
    cap.release()
    return info

def filter_player_detections(detections, min_confidence: float = 0.5, 
                           min_area: int = 1000) -> List:
    """Filter detections to keep only high-quality player detections."""
    filtered = []
    
    for detection in detections:
        bbox, confidence = detection
        x1, y1, x2, y2 = bbox
        
        # Calculate area
        area = (x2 - x1) * (y2 - y1)
        
        # Filter by confidence and area
        if confidence >= min_confidence and area >= min_area:
            filtered.append(detection)
    
    return filtered

def create_output_video(input_frames: List[np.ndarray], output_path: str, 
                       fps: float = 30.0):
    """Create output video from processed frames."""
    if not input_frames:
        return
        
    height, width = input_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in input_frames:
        out.write(frame)
    
    out.release()
    print(f"Output video saved to: {output_path}")

def log_processing_stats(frame_num: int, total_frames: int, num_detections: int, 
                        num_tracks: int):
    """Log processing statistics."""
    progress = (frame_num / total_frames) * 100
    print(f"Frame {frame_num:4d}/{total_frames} ({progress:5.1f}%) - "
          f"Detections: {num_detections:2d}, Active Tracks: {num_tracks:2d}")

def ensure_output_dir(output_dir: str):
    """Ensure output directory exists."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)