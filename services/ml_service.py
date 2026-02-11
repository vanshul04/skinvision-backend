"""
ML Service - Handles skin lesion classification using deep learning
"""

import numpy as np
from PIL import Image
import os
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    """
    Machine Learning service for skin lesion risk classification
    
    Uses a CNN model (EfficientNet/ResNet) to classify skin lesions
    and determine risk levels.
    """
    
    # Disease classes mapping
    DISEASE_CLASSES = {
        0: "Melanoma",
        1: "Melanocytic Nevus",
        2: "Benign Keratosis",
        3: "Basal Cell Carcinoma",
        4: "Actinic Keratosis",
        5: "Vascular Lesion",
        6: "Dermatofibroma"
    }
    
    # Risk mapping based on disease class
    RISK_MAPPING = {
        "Melanoma": "High",
        "Basal Cell Carcinoma": "High",
        "Actinic Keratosis": "Medium",
        "Melanocytic Nevus": "Low",
        "Benign Keratosis": "Low",
        "Vascular Lesion": "Low",
        "Dermatofibroma": "Low"
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML Service
        
        Args:
            model_path: Path to trained model file (optional)
        """
        self.model = None
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.is_loaded = False
        
        # Try to load model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(
                "No model file found. Using placeholder predictions. "
                "To use a real model, set MODEL_PATH environment variable."
            )
    
    def _load_model(self):
        """Load the trained TensorFlow/Keras model"""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_loaded
    
    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            image_array: Input image as numpy array
            
        Returns:
            Preprocessed image array
        """
        # Resize to model input size (typically 224x224 or 299x299)
        target_size = (224, 224)
        
        # Convert to PIL Image for resizing
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array)
        else:
            image = image_array
        
        # Resize and normalize
        image = image.resize(target_size)
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        if image_array.max() > 1:
            image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, image_array: np.ndarray) -> Dict:
        """
        Predict skin lesion class and risk level
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is not None:
            # Real model prediction
            predictions = self.model.predict(image_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
        else:
            # Placeholder prediction for development
            # Simulate realistic prediction distribution
            class_idx = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.1, 0.4, 0.2, 0.1, 0.1, 0.05, 0.05])
            confidence = np.random.uniform(0.65, 0.95)
        
        disease_class = self.DISEASE_CLASSES[class_idx]
        risk_level = self.RISK_MAPPING.get(disease_class, "Low")
        
        # Calculate risk score (0-1 scale)
        risk_score = self._calculate_risk_score(disease_class, confidence)
        
        # Generate explanation
        explanation = self._generate_explanation(disease_class, risk_level, confidence)
        
        return {
            "disease_class": disease_class,
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
            "risk_score": round(risk_score, 3),
            "explanation": explanation
        }
    
    def _calculate_risk_score(self, disease_class: str, confidence: float) -> float:
        """
        Calculate normalized risk score (0-1)
        
        Args:
            disease_class: Predicted disease class
            confidence: Model confidence
            
        Returns:
            Risk score between 0 and 1
        """
        base_risk = {
            "Melanoma": 0.9,
            "Basal Cell Carcinoma": 0.85,
            "Actinic Keratosis": 0.6,
            "Melanocytic Nevus": 0.2,
            "Benign Keratosis": 0.15,
            "Vascular Lesion": 0.1,
            "Dermatofibroma": 0.1
        }.get(disease_class, 0.5)
        
        # Adjust based on confidence
        risk_score = base_risk * confidence + (1 - confidence) * 0.3
        
        return min(risk_score, 1.0)
    
    def _generate_explanation(self, disease_class: str, risk_level: str, confidence: float) -> str:
        """
        Generate human-readable explanation
        
        Args:
            disease_class: Predicted disease class
            risk_level: Risk level (Low/Medium/High)
            confidence: Model confidence
            
        Returns:
            Explanation string
        """
        explanations = {
            "Melanoma": (
                f"The AI detected characteristics consistent with melanoma "
                f"(confidence: {confidence:.1f}%). This is a high-risk finding. "
                "Please consult a dermatologist immediately."
            ),
            "Basal Cell Carcinoma": (
                f"The analysis suggests possible basal cell carcinoma "
                f"(confidence: {confidence:.1f}%). This requires professional evaluation."
            ),
            "Actinic Keratosis": (
                f"Features consistent with actinic keratosis were detected "
                f"(confidence: {confidence:.1f}%). Regular monitoring is recommended."
            ),
            "Melanocytic Nevus": (
                f"The lesion appears to be a benign melanocytic nevus "
                f"(confidence: {confidence:.1f}%). Continue regular self-examinations."
            ),
            "Benign Keratosis": (
                f"The analysis indicates a benign keratosis "
                f"(confidence: {confidence:.1f}%). No immediate action required, "
                "but regular check-ups are recommended."
            ),
            "Vascular Lesion": (
                f"Features suggest a vascular lesion "
                f"(confidence: {confidence:.1f}%). Generally benign, but "
                "consultation is advised if it changes."
            ),
            "Dermatofibroma": (
                f"The lesion characteristics match dermatofibroma "
                f"(confidence: {confidence:.1f}%). This is typically benign."
            )
        }
        
        return explanations.get(
            disease_class,
            f"Analysis complete (confidence: {confidence:.1f}%). "
            "Please consult a dermatologist for professional evaluation."
        )

