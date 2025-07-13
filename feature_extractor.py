import cv2
import numpy as np
from typing import Tuple, List
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """Extract features from player detections for re-identification."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_appearance_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract appearance features from player crop."""
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512)  # Return zero vector for invalid bbox
            
        # Crop player region
        player_crop = image[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return np.zeros(512)
            
        # Color histogram features (RGB)
        color_features = []
        for i in range(3):
            hist = cv2.calcHist([player_crop], [i], None, [32], [0, 256])
            color_features.extend(hist.flatten())
        
        # Texture features using LBP-like approach
        gray_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        texture_features = self._extract_texture_features(gray_crop)
        
        # Shape features
        shape_features = self._extract_shape_features(player_crop)
        
        # Combine all features
        features = np.concatenate([color_features, texture_features, shape_features])
        
        # Normalize to prevent any feature from dominating
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract simple texture features."""
        if gray_image.size == 0:
            return np.zeros(64)
            
        # Resize for consistency
        gray_resized = cv2.resize(gray_image, (32, 64))
        
        # Gradient features
        grad_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        
        # Statistical features
        features = [
            np.mean(gray_resized), np.std(gray_resized),
            np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y)),
            np.std(grad_x), np.std(grad_y)
        ]
        
        # Add histogram of gradients
        hist_gradx = np.histogram(grad_x.flatten(), bins=16, range=(-100, 100))[0]
        hist_grady = np.histogram(grad_y.flatten(), bins=16, range=(-100, 100))[0]
        
        features.extend(hist_gradx)
        features.extend(hist_grady)
        
        # Add spatial information (divide image into grid)
        h, w = gray_resized.shape
        grid_features = []
        for i in range(4):
            for j in range(2):
                patch = gray_resized[i*h//4:(i+1)*h//4, j*w//2:(j+1)*w//2]
                if patch.size > 0:
                    grid_features.extend([np.mean(patch), np.std(patch)])
                else:
                    grid_features.extend([0, 0])
        
        features.extend(grid_features)
        
        return np.array(features)
    
    def _extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """Extract shape-based features."""
        if image.size == 0:
            return np.zeros(8)
            
        h, w = image.shape[:2]
        
        # Basic shape features
        aspect_ratio = w / (h + 1e-8)
        area = h * w
        
        # Convert to grayscale for contour analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate shape descriptors
            compactness = (perimeter * perimeter) / (contour_area + 1e-8)
            
            # Hu moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)
            else:
                hu_moments = np.zeros(7)
        else:
            contour_area = 0
            perimeter = 0
            compactness = 0
            hu_moments = np.zeros(7)
        
        shape_features = [aspect_ratio, area/10000, contour_area/10000, 
                         perimeter/100, compactness/100]
        
        return np.array(shape_features)
    
    def extract_positional_features(self, bbox: Tuple[int, int, int, int], 
                                  frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract position-based features."""
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape
        
        # Center coordinates (normalized)
        center_x = (x1 + x2) / (2 * frame_w)
        center_y = (y1 + y2) / (2 * frame_h)
        
        # Bounding box dimensions (normalized)
        width = (x2 - x1) / frame_w
        height = (y2 - y1) / frame_h
        
        # Position relative to frame regions
        left_region = center_x < 0.33
        center_region = 0.33 <= center_x <= 0.66
        right_region = center_x > 0.66
        
        top_region = center_y < 0.33
        middle_region = 0.33 <= center_y <= 0.66
        bottom_region = center_y > 0.66
        
        positional_features = [
            center_x, center_y, width, height,
            float(left_region), float(center_region), float(right_region),
            float(top_region), float(middle_region), float(bottom_region)
        ]
        
        return np.array(positional_features)
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute similarity between two feature vectors."""
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Convert to [0, 1] range
        similarity = (cosine_sim + 1) / 2
        
        return similarity