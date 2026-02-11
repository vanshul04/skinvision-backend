"""
Image upload routes for SkinVision AI
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import status

from services.image_service import ImageService
from database.mongo import MongoDatabase

router = APIRouter(prefix="/api", tags=["image-upload"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

image_service = ImageService()
db = MongoDatabase()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/upload-image",
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload a skin image, save it to disk, and store metadata in MongoDB.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only JPG and PNG images are allowed.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 5MB.",
        )

    # Optionally validate basic image correctness using ImageService
    if image_service.preprocess_image(raw_bytes) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image file.",
        )

    # Generate UUID filename while preserving extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        # Normalize to .jpg if extension missing or unusual
        ext = ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    relative_path = Path("uploads") / filename
    full_path = UPLOAD_DIR / filename

    try:
        with open(full_path, "wb") as out_file:
            out_file.write(raw_bytes)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded image.",
        )

    # Store metadata in MongoDB
    try:
        record_id = db.insert_image_metadata(
            image_path=str(relative_path).replace("\\", "/"),
            status="uploaded",
        )
    except Exception:
        # Best effort clean-up
        if full_path.exists():
            full_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store image metadata.",
        )

    return {
        "message": "Image uploaded successfully.",
        "image_path": str(relative_path).replace("\\", "/"),
        "record_id": record_id,
    }


