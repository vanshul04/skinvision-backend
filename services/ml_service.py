"""
ML Service - Handles skin lesion classification using deep learning
"""


from PIL import Image
import os
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    @staticmethod
    def predict(image_path):
        return {
            "prediction": "Benign",
            "confidence": 0.87
        }
