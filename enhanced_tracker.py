import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import math
from feature_extractor import FeatureExtractor
from tracking_config import TrackingConfig

@dataclass
class KalmanTracker:
    """Kalman filter for motion prediction."""
    def __init__(self, bbox: Tuple[int, int, int, int]):
        # State: [x, y, vx, vy] - center position and velocity
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        
        # Initialize with bbox center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kf.statePre = np.array([cx, cy, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([cx, cy, 0, 0], dtype=np.float32)
        
        self.bbox = bbox
        
    def predict(self):
        """Predict next position."""
        prediction = self.kf.predict()
        return prediction[:2]  # Return [x, y]
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update with new detection."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.correct(measurement)
        self.bbox = bbox
    
    def get_predicted_bbox(self) -> Tuple[int, int, int, int]:
        """Get predicted bounding box."""
        pred_center = self.predict()
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        x1 = int(pred_center[0] - w/2)
        y1 = int(pred_center[1] - h/2)
        x2 = int(pred_center[0] + w/2)
        y2 = int(pred_center[1] + h/2)
        return (x1, y1, x2, y2)

@dataclass
class EnhancedPlayerTrack:
    """Enhanced player track with temporal features and motion prediction."""
    id: int
    bbox: Tuple[int, int, int, int]
    features: np.ndarray
    last_seen_frame: int
    confidence: float
    total_detections: int
    missed_frames: int
    kalman_tracker: KalmanTracker
    feature_history: List[np.ndarray]
    bbox_history: List[Tuple[int, int, int, int]]
    confidence_history: List[float]
    
    def __post_init__(self):
        if not hasattr(self, 'feature_history'):
            self.feature_history = [self.features]
        if not hasattr(self, 'bbox_history'):
            self.bbox_history = [self.bbox]
        if not hasattr(self, 'confidence_history'):
            self.confidence_history = [self.confidence]
    
    def update(self, bbox: Tuple[int, int, int, int], features: np.ndarray, 
               frame_num: int, confidence: float):
        """Update track with temporal smoothing."""
        self.bbox = bbox
        self.last_seen_frame = frame_num
        self.confidence = confidence
        self.total_detections += 1
        self.missed_frames = 0
        
        # Update Kalman filter
        self.kalman_tracker.update(bbox)
        
        # Update feature history (keep last N features for temporal consistency)
        self.feature_history.append(features)
        if len(self.feature_history) > 10:  # Keep last 10 features
            self.feature_history.pop(0)
        
        # Update bbox history
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 20:  # Keep last 20 positions
            self.bbox_history.pop(0)
        
        # Update confidence history
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
        
        # Update features with temporal averaging
        self.features = self._compute_temporal_features()
    
    def _compute_temporal_features(self) -> np.ndarray:
        """Compute temporally averaged features."""
        if len(self.feature_history) == 1:
            return self.feature_history[0]
        
        # Weight recent features more heavily
        weights = np.exp(np.linspace(-1, 0, len(self.feature_history)))
        weights = weights / np.sum(weights)
        
        # Weighted average of features
        temporal_features = np.zeros_like(self.feature_history[0])
        for i, features in enumerate(self.feature_history):
            temporal_features += weights[i] * features
        
        return temporal_features
    
    def get_predicted_position(self) -> Tuple[int, int, int, int]:
        """Get motion-predicted position."""
        return self.kalman_tracker.get_predicted_bbox()
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        if len(self.bbox_history) < 2:
            return (0.0, 0.0)
        
        # Calculate velocity from recent positions
        recent_boxes = self.bbox_history[-5:]  # Last 5 positions
        if len(recent_boxes) < 2:
            return (0.0, 0.0)
        
        # Centers of recent boxes
        centers = []
        for box in recent_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            centers.append((cx, cy))
        
        # Average velocity
        vx = (centers[-1][0] - centers[0][0]) / len(centers)
        vy = (centers[-1][1] - centers[0][1]) / len(centers)
        
        return (vx, vy)
    
    def get_stability_score(self) -> float:
        """Get track stability score (higher = more stable)."""
        if len(self.confidence_history) < 3:
            return 0.5
        
        # Factors: detection consistency, confidence trend, motion smoothness
        confidence_std = np.std(self.confidence_history)
        confidence_mean = np.mean(self.confidence_history)
        
        # Motion smoothness
        motion_smoothness = 1.0
        if len(self.bbox_history) >= 3:
            velocities = []
            for i in range(1, len(self.bbox_history)):
                prev_center = ((self.bbox_history[i-1][0] + self.bbox_history[i-1][2])/2,
                              (self.bbox_history[i-1][1] + self.bbox_history[i-1][3])/2)
                curr_center = ((self.bbox_history[i][0] + self.bbox_history[i][2])/2,
                              (self.bbox_history[i][1] + self.bbox_history[i][3])/2)
                vel = math.sqrt((curr_center[0] - prev_center[0])**2 + 
                               (curr_center[1] - prev_center[1])**2)
                velocities.append(vel)
            
            if velocities:
                motion_smoothness = 1.0 / (1.0 + np.std(velocities))
        
        # Combined score
        stability = (confidence_mean * 0.4 + 
                    (1.0 - confidence_std) * 0.3 + 
                    motion_smoothness * 0.3)
        
        return min(1.0, max(0.0, stability))

class EnhancedPlayerTracker:
    """Enhanced tracker with Hungarian algorithm and motion prediction."""
    
    def __init__(self, config: TrackingConfig = None, 
                 max_missed_frames: int = None, 
                 similarity_threshold: float = None):
        """
        Initialize enhanced tracker with configuration.
        
        Args:
            config: TrackingConfig object with all parameters
            max_missed_frames: Override for max missed frames (deprecated, use config)
            similarity_threshold: Override for similarity threshold (deprecated, use config)
        """
        self.feature_extractor = FeatureExtractor()
        self.tracks: Dict[int, EnhancedPlayerTrack] = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Use config or defaults
        if config is None:
            config = TrackingConfig()
        
        # Allow backward compatibility with old parameters
        self.config = config
        self.max_missed_frames = max_missed_frames or config.MAX_MISSED_FRAMES
        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        self.max_distance_threshold = config.MAX_DISTANCE_THRESHOLD
        self.motion_weight = config.MOTION_WEIGHT
        self.appearance_weight = config.APPEARANCE_WEIGHT
        self.temporal_weight = config.TEMPORAL_WEIGHT
        self.min_track_length = config.MIN_TRACK_LENGTH
        
        # ID consistency enforcement
        self.id_consistency_buffer = {}  # track_id -> recent assignments
        
    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]], 
               frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Enhanced tracking update with Hungarian algorithm."""
        self.frame_count += 1
        
        # Extract features for all detections
        detection_features = []
        for bbox, conf in detections:
            features = self._extract_combined_features(frame, bbox)
            detection_features.append((bbox, conf, features))
        
        # Get predictions for existing tracks
        self._predict_track_positions()
        
        # Enhanced association with Hungarian algorithm
        matched_tracks, unmatched_detections, unmatched_tracks = \
            self._enhanced_association(detection_features)
        
        # Update matched tracks
        for track_id, (bbox, conf, features) in matched_tracks.items():
            self.tracks[track_id].update(bbox, features, self.frame_count, conf)
        
        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id].missed_frames += 1
        
        # Create new tracks for unmatched detections
        for bbox, conf, features in unmatched_detections:
            new_track = EnhancedPlayerTrack(
                id=self.next_id,
                bbox=bbox,
                features=features,
                last_seen_frame=self.frame_count,
                confidence=conf,
                total_detections=1,
                missed_frames=0,
                kalman_tracker=KalmanTracker(bbox),
                feature_history=[features],
                bbox_history=[bbox],
                confidence_history=[conf]
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Remove lost tracks
        self._remove_lost_tracks()
        
        # Return active tracks with stability filtering
        return self._get_stable_tracks()
    
    def _predict_track_positions(self):
        """Predict positions for all tracks using Kalman filters."""
        for track in self.tracks.values():
            track.kalman_tracker.predict()
    
    def _enhanced_association(self, detection_features: List) -> Tuple[Dict, List, List]:
        """Enhanced association using Hungarian algorithm and multiple cues."""
        if not self.tracks or not detection_features:
            return {}, detection_features, list(self.tracks.keys())
        
        track_ids = list(self.tracks.keys())
        
        # Build comprehensive cost matrix
        cost_matrix = self._build_cost_matrix(track_ids, detection_features)
        
        # Handle case where cost matrix is empty
        if cost_matrix.size == 0:
            return {}, detection_features, track_ids
        
        # Apply Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter assignments by threshold
        matched_tracks = {}
        matched_detection_indices = set()
        matched_track_indices = set()
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            cost = cost_matrix[track_idx, det_idx]
            similarity = 1.0 - cost  # Convert cost back to similarity
            
            if similarity > self.similarity_threshold:
                track_id = track_ids[track_idx]
                matched_tracks[track_id] = detection_features[det_idx]
                matched_track_indices.add(track_idx)
                matched_detection_indices.add(det_idx)
        
        # Unmatched detections and tracks
        unmatched_detections = [detection_features[i] for i in range(len(detection_features))
                              if i not in matched_detection_indices]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids))
                          if i not in matched_track_indices]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _build_cost_matrix(self, track_ids: List[int], 
                          detection_features: List) -> np.ndarray:
        """Build comprehensive cost matrix for Hungarian algorithm."""
        num_tracks = len(track_ids)
        num_detections = len(detection_features)
        
        if num_tracks == 0 or num_detections == 0:
            return np.array([])
        
        cost_matrix = np.full((num_tracks, num_detections), 1.0, dtype=np.float32)
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            
            for j, (det_bbox, det_conf, det_features) in enumerate(detection_features):
                # 1. Appearance similarity
                app_similarity = self.feature_extractor.compute_similarity(
                    track.features, det_features)
                
                # 2. Position similarity (current and predicted)
                current_pos_sim = self._compute_position_similarity(track.bbox, det_bbox)
                predicted_pos_sim = self._compute_position_similarity(
                    track.get_predicted_position(), det_bbox)
                pos_similarity = max(current_pos_sim, predicted_pos_sim)
                
                # 3. Motion consistency
                motion_similarity = self._compute_motion_consistency(track, det_bbox)
                
                # 4. Size consistency
                size_similarity = self._compute_size_similarity(track.bbox, det_bbox)
                
                # 5. Temporal consistency
                temporal_similarity = self._compute_temporal_consistency(track, det_features)
                
                # Combined similarity with adaptive weights
                stability_score = track.get_stability_score()
                
                # Adjust weights based on track stability
                adj_appearance_weight = self.appearance_weight * (1.0 + stability_score * 0.5)
                adj_motion_weight = self.motion_weight * (1.0 + stability_score * 0.3)
                
                total_similarity = (
                    adj_appearance_weight * app_similarity +
                    adj_motion_weight * pos_similarity +
                    0.15 * motion_similarity +
                    0.1 * size_similarity +
                    self.temporal_weight * temporal_similarity
                )
                
                # Distance gating - reject if too far
                distance = self._compute_center_distance(track.bbox, det_bbox)
                if distance > self.max_distance_threshold:
                    total_similarity *= 0.1  # Heavily penalize distant matches
                
                # Convert similarity to cost (Hungarian minimizes cost)
                cost_matrix[i, j] = 1.0 - total_similarity
        
        return cost_matrix
    
    def _compute_motion_consistency(self, track: EnhancedPlayerTrack, 
                                  det_bbox: Tuple[int, int, int, int]) -> float:
        """Compute motion consistency between track and detection."""
        if len(track.bbox_history) < 2:
            return 0.5  # Neutral score for new tracks
        
        # Get track velocity
        vx, vy = track.get_velocity()
        
        # Predict where track should be
        current_center = ((track.bbox[0] + track.bbox[2])/2, 
                         (track.bbox[1] + track.bbox[3])/2)
        predicted_center = (current_center[0] + vx, current_center[1] + vy)
        
        # Compare with detection center
        det_center = ((det_bbox[0] + det_bbox[2])/2, 
                     (det_bbox[1] + det_bbox[3])/2)
        
        distance = math.sqrt((predicted_center[0] - det_center[0])**2 + 
                           (predicted_center[1] - det_center[1])**2)
        
        # Convert distance to similarity
        max_motion_distance = 50  # pixels
        similarity = max(0, 1.0 - (distance / max_motion_distance))
        
        return similarity
    
    def _compute_size_similarity(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """Compute size similarity between bounding boxes."""
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        
        area1, area2 = w1 * h1, w2 * h2
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # IoU-like similarity for sizes
        intersection = min(area1, area2)
        union = max(area1, area2)
        
        return intersection / union
    
    def _compute_temporal_consistency(self, track: EnhancedPlayerTrack, 
                                    det_features: np.ndarray) -> float:
        """Compute temporal consistency of features."""
        if len(track.feature_history) < 2:
            return 0.5
        
        # Compare with recent features
        similarities = []
        for hist_features in track.feature_history[-3:]:  # Last 3 features
            sim = self.feature_extractor.compute_similarity(hist_features, det_features)
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _compute_center_distance(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """Compute Euclidean distance between bbox centers."""
        center1 = ((bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2)
        center2 = ((bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _compute_position_similarity(self, bbox1: Tuple[int, int, int, int], 
                                   bbox2: Tuple[int, int, int, int]) -> float:
        """Enhanced position similarity."""
        distance = self._compute_center_distance(bbox1, bbox2)
        
        # Adaptive threshold based on bbox size
        avg_size = ((bbox1[2] - bbox1[0]) + (bbox1[3] - bbox1[1]) + 
                   (bbox2[2] - bbox2[0]) + (bbox2[3] - bbox2[1])) / 4
        adaptive_threshold = max(50, avg_size * 0.5)
        
        similarity = max(0, 1.0 - (distance / adaptive_threshold))
        return similarity
    
    def _extract_combined_features(self, frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract enhanced combined features."""
        appearance_features = self.feature_extractor.extract_appearance_features(frame, bbox)
        positional_features = self.feature_extractor.extract_positional_features(
            bbox, frame.shape[:2])
        
        # Normalize features
        appearance_features = appearance_features / (np.linalg.norm(appearance_features) + 1e-8)
        positional_features = positional_features / (np.linalg.norm(positional_features) + 1e-8)
        
        # Combine with optimized weights
        combined_features = np.concatenate([
            appearance_features * 0.8,
            positional_features * 0.2
        ])
        
        return combined_features
    
    def _get_stable_tracks(self) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Return only stable tracks to reduce ID switching."""
        active_tracks = []
        
        for track in self.tracks.values():
            if track.missed_frames == 0:  # Currently detected
                # Only include tracks that have been seen enough times
                if track.total_detections >= self.min_track_length or track.get_stability_score() > 0.7:
                    active_tracks.append((track.id, track.bbox, track.confidence))
        
        return active_tracks
    
    def _remove_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.missed_frames > self.max_missed_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_track_history(self) -> Dict[int, Dict]:
        """Get enhanced track history."""
        history = {}
        for track_id, track in self.tracks.items():
            history[track_id] = {
                'total_detections': track.total_detections,
                'last_seen_frame': track.last_seen_frame,
                'missed_frames': track.missed_frames,
                'confidence': track.confidence,
                'stability_score': track.get_stability_score(),
                'track_length': len(track.bbox_history)
            }
        return history
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.id_consistency_buffer.clear()