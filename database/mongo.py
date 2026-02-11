"""
MongoDB connection utilities for SkinVision AI
"""

import os
from typing import Any, Dict
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection

load_dotenv()


class MongoDatabase:
    """
    MongoDB wrapper for image metadata storage.
    """

    def __init__(self) -> None:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("MONGO_DB_NAME", "skinvision_ai")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.image_reports: Collection = self.db["image_reports"]

    def insert_image_metadata(
        self,
        image_path: str,
        status: str = "uploaded",
    ) -> str:
        """
        Insert image metadata document and return inserted ID as string.
        """
        doc: Dict[str, Any] = {
            "image_path": image_path,
            "uploaded_at": datetime.utcnow(),
            "status": status,
        }
        result = self.image_reports.insert_one(doc)
        return str(result.inserted_id)


