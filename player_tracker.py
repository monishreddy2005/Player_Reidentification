import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from feature_extractor import FeatureExtractor

@dataclass
class PlayerTrack:
    """Represents a player track with ID and features."""
    id: int
    bbox: Tuple[int, int, int, int]
    features: np.ndarray
    last_seen_frame: int
    confidence: float
    total_detections: int
    missed_frames: int
    
    def update(self, bbox: Tuple[int, int, int, int], features: np.ndarray, 
               frame_num: int, confidence: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.features = features
        self.last_seen_frame = frame_num
        self.confidence = confidence
        self.total_detections += 1
        self.missed_frames = 0

class PlayerTracker:
    """Main tracker for player re-identification."""
    
    def __init__(self, max_missed_frames: int = 30, similarity_threshold: float = 0.6):
        self.feature_extractor = FeatureExtractor()
        self.tracks: Dict[int, PlayerTrack] = {}
        self.next_id = 1
        self.max_missed_frames = max_missed_frames
        self.similarity_threshold = similarity_threshold
        self.frame_count = 0
        
        # For temporal consistency
        self.position_threshold = 100  # pixels
        self.appearance_weight = 0.7
        self.position_weight = 0.3
        
    def update(self, detections: List[Tuple[Tuple[int, int, int, int], float]], 
               frame: np.ndarray) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, confidence) tuples
            frame: Current frame image
            
        Returns:
            List of (track_id, bbox, confidence) tuples
        """
        self.frame_count += 1
        
        # Extract features for all detections
        detection_features = []
        for bbox, conf in detections:
            features = self._extract_combined_features(frame, bbox)
            detection_features.append((bbox, conf, features))
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detection_features)
        
        # Update matched tracks
        for track_id, (bbox, conf, features) in matched_tracks.items():
            self.tracks[track_id].update(bbox, features, self.frame_count, conf)
        
        # Handle unmatched tracks (increment missed frames)
        for track_id in unmatched_tracks:
            self.tracks[track_id].missed_frames += 1
        
        # Create new tracks for unmatched detections
        for bbox, conf, features in unmatched_detections:
            new_track = PlayerTrack(
                id=self.next_id,
                bbox=bbox,
                features=features,
                last_seen_frame=self.frame_count,
                confidence=conf,
                total_detections=1,
                missed_frames=0
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1
        
        # Remove tracks that have been missed for too long
        self._remove_lost_tracks()
        
        # Return current active tracks
        active_tracks = []
        for track in self.tracks.values():
            if track.missed_frames == 0:  # Only return tracks with current detections
                active_tracks.append((track.id, track.bbox, track.confidence))
        
        return active_tracks
    
    def _extract_combined_features(self, frame: np.ndarray, 
                                 bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract combined appearance and positional features."""
        appearance_features = self.feature_extractor.extract_appearance_features(frame, bbox)
        positional_features = self.feature_extractor.extract_positional_features(
            bbox, frame.shape[:2])
        
        # Combine features with weights
        combined_features = np.concatenate([
            appearance_features * self.appearance_weight,
            positional_features * self.position_weight
        ])
        
        return combined_features
    
    def _associate_detections_to_tracks(self, detection_features: List) -> Tuple[Dict, List, List]:
        """Associate detections to existing tracks using Hungarian algorithm approach."""
        if not self.tracks or not detection_features:
            return {}, detection_features, list(self.tracks.keys())
        
        # Compute similarity matrix
        track_ids = list(self.tracks.keys())
        similarity_matrix = np.zeros((len(track_ids), len(detection_features)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, (bbox, conf, features) in enumerate(detection_features):
                # Appearance similarity
                app_similarity = self.feature_extractor.compute_similarity(
                    track.features, features)
                
                # Position similarity (distance-based)
                pos_similarity = self._compute_position_similarity(track.bbox, bbox)
                
                # Combined similarity
                total_similarity = (self.appearance_weight * app_similarity + 
                                  self.position_weight * pos_similarity)
                
                similarity_matrix[i, j] = total_similarity
        
        # Simple greedy matching (can be improved with Hungarian algorithm)
        matched_tracks = {}
        matched_detection_indices = set()
        matched_track_indices = set()
        
        # Sort by similarity and match greedily
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detection_features)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    matches.append((similarity_matrix[i, j], i, j))
        
        matches.sort(reverse=True)  # Sort by similarity (highest first)
        
        for similarity, track_idx, det_idx in matches:
            if track_idx not in matched_track_indices and det_idx not in matched_detection_indices:
                track_id = track_ids[track_idx]
                matched_tracks[track_id] = detection_features[det_idx]
                matched_track_indices.add(track_idx)
                matched_detection_indices.add(det_idx)
        
        # Unmatched detections
        unmatched_detections = [detection_features[i] for i in range(len(detection_features))
                              if i not in matched_detection_indices]
        
        # Unmatched tracks
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids))
                          if i not in matched_track_indices]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _compute_position_similarity(self, bbox1: Tuple[int, int, int, int], 
                                   bbox2: Tuple[int, int, int, int]) -> float:
        """Compute position-based similarity between two bounding boxes."""
        # Calculate center points
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Euclidean distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Convert distance to similarity (closer = higher similarity)
        max_distance = self.position_threshold
        similarity = max(0, 1 - (distance / max_distance))
        
        return similarity
    
    def _remove_lost_tracks(self):
        """Remove tracks that have been lost for too many frames."""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.missed_frames > self.max_missed_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_track_history(self) -> Dict[int, Dict]:
        """Get history of all tracks for analysis."""
        history = {}
        for track_id, track in self.tracks.items():
            history[track_id] = {
                'total_detections': track.total_detections,
                'last_seen_frame': track.last_seen_frame,
                'missed_frames': track.missed_frames,
                'confidence': track.confidence
            }
        return history
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0