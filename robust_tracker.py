import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from feature_extractor import FeatureExtractor

@dataclass
class SimpleKalmanTracker:
    """Simplified, more stable Kalman filter for motion prediction."""
    def __init__(self, bbox: Tuple[int, int, int, int]):
        # Simple 2D position tracking [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix (we measure position only)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], dtype=np.float32)
        
        # Transition matrix (position + velocity model)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], dtype=np.float32)
        
        # Conservative noise parameters for stability
        self.kf.processNoiseCov = 0.01 * np.eye(4, dtype=np.float32)  # Very low process noise
        self.kf.measurementNoiseCov = 0.5 * np.eye(2, dtype=np.float32)  # Higher measurement noise
        
        # Initialize with center of bbox
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Initialize state
        self.kf.statePre = np.array([cx, cy, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([cx, cy, 0, 0], dtype=np.float32)
        
        self.last_bbox = bbox
        self.predicted_position = (cx, cy)
        
    def predict(self) -> Tuple[float, float]:
        """Predict next position."""
        try:
            prediction = self.kf.predict()
            self.predicted_position = (float(prediction[0]), float(prediction[1]))
            return self.predicted_position
        except:
            # Fallback to last known position
            cx = (self.last_bbox[0] + self.last_bbox[2]) / 2
            cy = (self.last_bbox[1] + self.last_bbox[3]) / 2
            return (cx, cy)
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update with new detection."""
        try:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            self.kf.correct(measurement)
            self.last_bbox = bbox
        except:
            # If Kalman update fails, just store the bbox
            self.last_bbox = bbox

@dataclass
class RobustPlayerTrack:
    """Robust player track with conservative features."""
    id: int
    bbox: Tuple[int, int, int, int]
    features: np.ndarray
    last_seen_frame: int
    confidence: float
    total_detections: int
    missed_frames: int
    kalman_tracker: SimpleKalmanTracker
    
    # Conservative feature history
    recent_features: List[np.ndarray]
    recent_positions: List[Tuple[float, float]]
    recent_confidences: List[float]
    
    # Stability metrics
    position_variance: float = 0.0
    confidence_trend: float = 1.0
    
    def __post_init__(self):
        if not hasattr(self, 'recent_features'):
            self.recent_features = [self.features]
        if not hasattr(self, 'recent_positions'):
            cx = (self.bbox[0] + self.bbox[2]) / 2
            cy = (self.bbox[1] + self.bbox[3]) / 2
            self.recent_positions = [(cx, cy)]
        if not hasattr(self, 'recent_confidences'):
            self.recent_confidences = [self.confidence]
    
    def update(self, bbox: Tuple[int, int, int, int], features: np.ndarray, 
               frame_num: int, confidence: float):
        """Conservative update with stability tracking."""
        self.bbox = bbox
        self.last_seen_frame = frame_num
        self.confidence = confidence
        self.total_detections += 1
        self.missed_frames = 0
        
        # Update Kalman filter
        self.kalman_tracker.update(bbox)
        
        # Update recent history (keep small window for stability)
        self.recent_features.append(features)
        if len(self.recent_features) > 5:  # Only keep last 5
            self.recent_features.pop(0)
        
        # Update position history
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.recent_positions.append((cx, cy))
        if len(self.recent_positions) > 5:
            self.recent_positions.pop(0)
        
        # Update confidence history
        self.recent_confidences.append(confidence)
        if len(self.recent_confidences) > 5:
            self.recent_confidences.pop(0)
        
        # Update stability metrics
        self._update_stability_metrics()
        
        # Conservative feature update (average of recent features)
        self.features = self._get_stable_features()
    
    def _update_stability_metrics(self):
        """Update stability metrics for this track."""
        if len(self.recent_positions) >= 3:
            # Position variance (lower = more stable)
            positions = np.array(self.recent_positions)
            self.position_variance = np.var(positions)
        
        if len(self.recent_confidences) >= 3:
            # Confidence trend (stable = close to 1.0)
            confidences = np.array(self.recent_confidences)
            self.confidence_trend = 1.0 - np.std(confidences)
    
    def _get_stable_features(self) -> np.ndarray:
        """Get stable features by averaging recent ones."""
        if len(self.recent_features) == 1:
            return self.recent_features[0]
        
        # Simple average of recent features for stability
        return np.mean(self.recent_features, axis=0)
    
    def get_predicted_position(self) -> Tuple[float, float]:
        """Get predicted position from Kalman filter."""
        return self.kalman_tracker.predict()
    
    def get_stability_score(self) -> float:
        """Get overall stability score (0-1, higher = more stable)."""
        base_score = 0.5
        
        # Factor in detection count (more detections = more stable)
        detection_score = min(1.0, self.total_detections / 10.0)
        
        # Factor in confidence trend
        confidence_score = max(0.0, self.confidence_trend)
        
        # Factor in position stability (lower variance = higher score)
        position_score = max(0.0, 1.0 - (self.position_variance / 10000.0))
        
        # Combined score
        stability = (base_score * 0.2 + 
                    detection_score * 0.4 + 
                    confidence_score * 0.2 + 
                    position_score * 0.2)
        
        return min(1.0, max(0.0, stability))

class RobustPlayerTracker:
    """Robust, conservative tracker focused on ID stability."""
    
    def __init__(self, debug_mode: bool = True):
        """Initialize robust tracker with conservative settings."""
        self.feature_extractor = FeatureExtractor()
        self.tracks: Dict[int, RobustPlayerTrack] = {}
        self.next_id = 1
        self.frame_count = 0
        self.debug_mode = debug_mode
        
        # Conservative parameters (prioritize stability over complexity)
        self.similarity_threshold = 0.7  # High threshold for conservative matching
        self.max_distance_threshold = 200  # Generous distance allowance
        self.max_missed_frames = 30
        self.min_track_length = 3  # Shorter minimum for responsiveness
        
        # Simple feature weights (heavily favor appearance)
        self.appearance_weight = 0.8
        self.position_weight = 0.2
        
        # ID consistency tracking
        self.id_switches = 0
        self.total_assignments = 0
        
        if self.debug_mode:
            print("🛡️ Robust Tracker initialized with conservative settings:")
            print(f"   Similarity threshold: {self.similarity_threshold}")
            print(f"   Max distance: {self.max_distance_threshold}px")
            print(f"   Appearance weight: {self.appearance_weight}")
            print(f"   Position weight: {self.position_weight}")
    
    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]], 
               frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Conservative tracking update."""
        self.frame_count += 1
        
        if self.debug_mode and self.frame_count % 30 == 0:
            print(f"📊 Frame {self.frame_count}: {len(detections)} detections, {len(self.tracks)} active tracks")
        
        # Extract features for all detections
        detection_features = []
        for bbox, conf in detections:
            features = self._extract_robust_features(frame, bbox)
            detection_features.append((bbox, conf, features))
        
        # Conservative association (prioritize existing tracks)
        matched_tracks, unmatched_detections, unmatched_tracks = \
            self._conservative_association(detection_features)
        
        # Update matched tracks
        for track_id, (bbox, conf, features) in matched_tracks.items():
            self.tracks[track_id].update(bbox, features, self.frame_count, conf)
            self.total_assignments += 1
        
        # Handle unmatched tracks (be patient)
        for track_id in unmatched_tracks:
            self.tracks[track_id].missed_frames += 1
        
        # Create new tracks only for stable detections
        for bbox, conf, features in unmatched_detections:
            if conf >= 0.6:  # Only create tracks for high-confidence detections
                new_track = RobustPlayerTrack(
                    id=self.next_id,
                    bbox=bbox,
                    features=features,
                    last_seen_frame=self.frame_count,
                    confidence=conf,
                    total_detections=1,
                    missed_frames=0,
                    kalman_tracker=SimpleKalmanTracker(bbox),
                    recent_features=[features],
                    recent_positions=[(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2],
                    recent_confidences=[conf]
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1
                
                if self.debug_mode:
                    print(f"✨ Created new track ID {self.next_id - 1} (conf: {conf:.2f})")
        
        # Remove lost tracks
        self._remove_lost_tracks()
        
        # Return stable tracks only
        return self._get_stable_tracks()
    
    def _extract_robust_features(self, frame: np.ndarray, 
                                bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract robust, stable features."""
        try:
            # Get basic appearance features
            appearance_features = self.feature_extractor.extract_appearance_features(frame, bbox)
            positional_features = self.feature_extractor.extract_positional_features(
                bbox, frame.shape[:2])
            
            # Simple normalization (prevent over-normalization issues)
            if np.linalg.norm(appearance_features) > 0:
                appearance_features = appearance_features / np.linalg.norm(appearance_features)
            else:
                appearance_features = np.zeros_like(appearance_features)
            
            if np.linalg.norm(positional_features) > 0:
                positional_features = positional_features / np.linalg.norm(positional_features)
            else:
                positional_features = np.zeros_like(positional_features)
            
            # Conservative combination (heavily favor appearance)
            combined_features = np.concatenate([
                appearance_features * self.appearance_weight,
                positional_features * self.position_weight
            ])
            
            return combined_features
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Feature extraction error: {e}")
            # Return zero features on error
            return np.zeros(100)  # Fallback feature size
    
    def _conservative_association(self, detection_features: List) -> Tuple[Dict, List, List]:
        """Conservative association that prioritizes ID stability."""
        if not self.tracks or not detection_features:
            return {}, detection_features, list(self.tracks.keys())
        
        track_ids = list(self.tracks.keys())
        matched_tracks = {}
        matched_detection_indices = set()
        matched_track_indices = set()
        
        # Simple distance-first matching for stability
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            best_match_idx = -1
            best_similarity = 0.0
            best_distance = float('inf')
            
            for j, (det_bbox, det_conf, det_features) in enumerate(detection_features):
                if j in matched_detection_indices:
                    continue
                
                # Primary filter: distance check
                distance = self._compute_distance(track.bbox, det_bbox)
                if distance > self.max_distance_threshold:
                    continue
                
                # Secondary filter: appearance similarity
                try:
                    similarity = self.feature_extractor.compute_similarity(
                        track.features, det_features)
                except:
                    similarity = 0.0
                
                # Conservative matching: prioritize close, similar detections
                if similarity > self.similarity_threshold:
                    if distance < best_distance:  # Prefer closer detections
                        best_match_idx = j
                        best_similarity = similarity
                        best_distance = distance
            
            # Assign best match if found
            if best_match_idx >= 0:
                matched_tracks[track_id] = detection_features[best_match_idx]
                matched_detection_indices.add(best_match_idx)
                matched_track_indices.add(i)
                
                if self.debug_mode and self.frame_count % 30 == 0:
                    print(f"🔗 Track {track_id} matched (sim: {best_similarity:.2f}, dist: {best_distance:.1f})")
        
        # Unmatched items
        unmatched_detections = [detection_features[i] for i in range(len(detection_features))
                              if i not in matched_detection_indices]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids))
                          if i not in matched_track_indices]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _compute_distance(self, bbox1: Tuple[int, int, int, int], 
                         bbox2: Tuple[int, int, int, int]) -> float:
        """Compute simple Euclidean distance between bbox centers."""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _get_stable_tracks(self) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Return only stable, reliable tracks."""
        stable_tracks = []
        
        for track in self.tracks.values():
            if track.missed_frames == 0:  # Currently detected
                # Only show tracks that meet minimum stability criteria
                if (track.total_detections >= self.min_track_length and 
                    track.get_stability_score() > 0.3):
                    stable_tracks.append((track.id, track.bbox, track.confidence))
        
        return stable_tracks
    
    def _remove_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.missed_frames > self.max_missed_frames:
                tracks_to_remove.append(track_id)
        
        if tracks_to_remove and self.debug_mode:
            print(f"🗑️ Removing {len(tracks_to_remove)} lost tracks: {tracks_to_remove}")
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_track_history(self) -> Dict[int, Dict]:
        """Get tracking history with stability metrics."""
        history = {}
        for track_id, track in self.tracks.items():
            history[track_id] = {
                'total_detections': track.total_detections,
                'last_seen_frame': track.last_seen_frame,
                'missed_frames': track.missed_frames,
                'confidence': track.confidence,
                'stability_score': track.get_stability_score(),
                'position_variance': track.position_variance
            }
        return history
    
    def get_stats(self) -> Dict:
        """Get tracking statistics for debugging."""
        id_switch_rate = (self.id_switches / max(1, self.total_assignments)) * 100
        return {
            'total_tracks_created': self.next_id - 1,
            'active_tracks': len(self.tracks),
            'id_switches': self.id_switches,
            'total_assignments': self.total_assignments,
            'id_switch_rate': id_switch_rate,
            'frames_processed': self.frame_count
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.id_switches = 0
        self.total_assignments = 0
        
        if self.debug_mode:
            print("🔄 Tracker reset")

class HybridTracker:
    """Hybrid tracker that can switch between different tracking strategies."""
    
    def __init__(self, mode: str = "robust"):
        """Initialize hybrid tracker."""
        self.mode = mode
        self.tracker = None
        self._initialize_tracker()
        
    def _initialize_tracker(self):
        """Initialize the appropriate tracker based on mode."""
        if self.mode == "robust":
            self.tracker = RobustPlayerTracker(debug_mode=True)
        elif self.mode == "basic":
            from player_tracker import PlayerTracker
            self.tracker = PlayerTracker(
                max_missed_frames=30,
                similarity_threshold=0.6
            )
        else:
            # Default to robust
            self.tracker = RobustPlayerTracker(debug_mode=True)
        
        print(f"🎯 Initialized {self.mode} tracker")
    
    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]], 
               frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """Update using the current tracker."""
        return self.tracker.update(detections, frame)
    
    def get_track_history(self) -> Dict[int, Dict]:
        """Get track history."""
        return self.tracker.get_track_history()
    
    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        if hasattr(self.tracker, 'get_stats'):
            return self.tracker.get_stats()
        else:
            return {'mode': self.mode, 'stats': 'not_available'}
    
    def reset(self):
        """Reset tracker."""
        self.tracker.reset()