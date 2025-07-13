"""
Configuration file for enhanced player tracking system.
Adjust these parameters based on your specific use case.
"""

class TrackingConfig:
    """Configuration class for tracking parameters."""
    
    # === SIMILARITY THRESHOLDS ===
    # Lower values = more strict matching (less ID switching, but may lose tracks)
    # Higher values = more lenient matching (may cause ID switching)
    SIMILARITY_THRESHOLD = 0.4  # Main similarity threshold (0.3-0.7)
    
    # === DISTANCE THRESHOLDS ===
    # Maximum distance between detections to consider matching (pixels)
    MAX_DISTANCE_THRESHOLD = 150  # For close player scenarios, reduce to 100-120
    
    # === TRACKING PERSISTENCE ===
    MAX_MISSED_FRAMES = 30  # How long to keep tracks without detections
    MIN_TRACK_LENGTH = 5    # Minimum detections before showing stable ID
    
    # === FEATURE WEIGHTS ===
    # Adjust based on your video characteristics
    APPEARANCE_WEIGHT = 0.5    # Visual appearance similarity (0.4-0.7)
    MOTION_WEIGHT = 0.3        # Position and motion consistency (0.2-0.4)
    TEMPORAL_WEIGHT = 0.2      # Temporal feature consistency (0.1-0.3)
    
    # === MOTION PREDICTION ===
    # Kalman filter noise parameters
    PROCESS_NOISE = 0.03       # Lower = smoother tracking, Higher = more responsive
    MEASUREMENT_NOISE = 0.1    # Lower = trust measurements more
    
    # === FEATURE EXTRACTION ===
    # Number of historical features to keep for temporal consistency
    FEATURE_HISTORY_LENGTH = 10
    BBOX_HISTORY_LENGTH = 20
    CONFIDENCE_HISTORY_LENGTH = 10
    
    # === SCENARIO-SPECIFIC PRESETS ===
    
    @classmethod
    def for_crowded_scenes(cls):
        """Configuration optimized for crowded scenes with many close players."""
        config = cls()
        config.SIMILARITY_THRESHOLD = 0.5      # More lenient for crowded scenes
        config.MAX_DISTANCE_THRESHOLD = 100    # Shorter distance for close players
        config.APPEARANCE_WEIGHT = 0.6          # Rely more on appearance
        config.MOTION_WEIGHT = 0.25             # Less on motion (players move erratically)
        config.TEMPORAL_WEIGHT = 0.15
        config.MIN_TRACK_LENGTH = 7             # More stable tracks
        return config
    
    @classmethod
    def for_sparse_scenes(cls):
        """Configuration optimized for scenes with few, well-separated players."""
        config = cls()
        config.SIMILARITY_THRESHOLD = 0.3      # Stricter matching
        config.MAX_DISTANCE_THRESHOLD = 200    # Longer distance OK
        config.APPEARANCE_WEIGHT = 0.4          # Balance appearance and motion
        config.MOTION_WEIGHT = 0.4
        config.TEMPORAL_WEIGHT = 0.2
        config.MIN_TRACK_LENGTH = 3             # Show IDs sooner
        return config
    
    @classmethod
    def for_fast_motion(cls):
        """Configuration optimized for fast-moving sports."""
        config = cls()
        config.SIMILARITY_THRESHOLD = 0.4
        config.MAX_DISTANCE_THRESHOLD = 180    # Allow larger movements
        config.APPEARANCE_WEIGHT = 0.4
        config.MOTION_WEIGHT = 0.4              # Higher motion weight
        config.TEMPORAL_WEIGHT = 0.2
        config.PROCESS_NOISE = 0.05             # More responsive tracking
        config.MAX_MISSED_FRAMES = 20           # Shorter persistence for fast sports
        return config
    
    @classmethod
    def for_occlusion_heavy(cls):
        """Configuration optimized for scenes with frequent occlusions."""
        config = cls()
        config.SIMILARITY_THRESHOLD = 0.35     # Stricter when re-appearing
        config.MAX_DISTANCE_THRESHOLD = 130
        config.APPEARANCE_WEIGHT = 0.6          # Rely heavily on appearance
        config.MOTION_WEIGHT = 0.2
        config.TEMPORAL_WEIGHT = 0.2
        config.MAX_MISSED_FRAMES = 45           # Keep tracks longer during occlusions
        config.MIN_TRACK_LENGTH = 8             # Very stable before showing
        return config

# === USAGE EXAMPLES ===
"""
# Use default configuration
config = TrackingConfig()

# Use preset for crowded scenes
config = TrackingConfig.for_crowded_scenes()

# Custom configuration
config = TrackingConfig()
config.SIMILARITY_THRESHOLD = 0.45
config.APPEARANCE_WEIGHT = 0.7
"""

# === PARAMETER TUNING GUIDE ===
"""
If you're experiencing:

1. ID SWITCHING when players come close:
   - Lower SIMILARITY_THRESHOLD (0.3-0.4)
   - Increase APPEARANCE_WEIGHT (0.6-0.7)
   - Decrease MAX_DISTANCE_THRESHOLD (100-120)
   - Increase MIN_TRACK_LENGTH (6-8)

2. LOST TRACKS (IDs disappearing):
   - Increase SIMILARITY_THRESHOLD (0.5-0.6)
   - Increase MAX_DISTANCE_THRESHOLD (150-200)
   - Increase MAX_MISSED_FRAMES (40-50)
   - Decrease MIN_TRACK_LENGTH (3-4)

3. TOO MANY FALSE IDs:
   - Increase MIN_TRACK_LENGTH (6-10)
   - Lower SIMILARITY_THRESHOLD (0.3-0.4)
   - Increase APPEARANCE_WEIGHT

4. POOR TRACKING in specific scenarios:
   - Crowded scenes: Use for_crowded_scenes()
   - Fast motion: Use for_fast_motion()
   - Many occlusions: Use for_occlusion_heavy()
   - Few players: Use for_sparse_scenes()
"""