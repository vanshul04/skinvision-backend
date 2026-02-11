"""
Image Service - Handles image preprocessing and validation
"""

import numpy as np
from PIL import Image
import io
import cv2
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageService:
    """
    Service for image preprocessing and quality validation
    """
    
    MIN_RESOLUTION = (224, 224)  # Minimum resolution for analysis
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size
    
    def __init__(self):
        """Initialize Image Service"""
        pass
    
    def preprocess_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Preprocess uploaded image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Preprocessed image array or None if invalid
        """
        try:
            # Validate file size
            if len(image_data) > self.MAX_FILE_SIZE:
                logger.warning(f"Image file too large: {len(image_data)} bytes")
                return None
            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Validate resolution
            width, height = image.size
            if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                logger.warning(
                    f"Image resolution too low: {width}x{height}. "
                    f"Minimum: {self.MIN_RESOLUTION}"
                )
                # Still process, but warn
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Optional: Check for blur (using Laplacian variance)
            blur_score = self._check_blur(image_array)
            if blur_score < 100:  # Threshold for blur detection
                logger.warning(f"Image may be blurry (blur score: {blur_score})")
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _check_blur(self, image_array: np.ndarray) -> float:
        """
        Check image blur using Laplacian variance
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Blur score (higher = sharper)
        """
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var
        except Exception as e:
            logger.warning(f"Blur check failed: {e}")
            return 1000.0  # Assume sharp if check fails
    
    def validate_image_quality(
        self,
        image_array: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate image quality for analysis
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        warnings = []
        
        # Check resolution
        height, width = image_array.shape[:2]
        if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
            warnings.append(
                f"Low resolution ({width}x{height}). "
                "Higher resolution images provide better accuracy."
            )
        
        # Check blur
        blur_score = self._check_blur(image_array)
        if blur_score < 100:
            warnings.append("Image appears blurry. Please use a sharper image.")
        
        # Check brightness
        mean_brightness = np.mean(image_array)
        if mean_brightness < 50:
            warnings.append("Image is too dark. Ensure good lighting.")
        elif mean_brightness > 200:
            warnings.append("Image is overexposed. Reduce lighting.")
        
        if warnings:
            return True, " | ".join(warnings)
        
        return True, None

