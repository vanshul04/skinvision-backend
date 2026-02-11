"""
SkinVision AI Backend - FastAPI Application
Main entry point for the skin lesion analysis API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from dotenv import load_dotenv

from services.ml_service import MLService
from services.image_service import ImageService
from routes.upload import router as upload_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SkinVision AI API",
    description="AI-powered skin lesion risk classification API",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MONGO_URI = os.getenv("MONGO_URI")

# Initialize services
ml_service = MLService()
image_service = ImageService()

# Include routers
app.include_router(upload_router)


class AnalysisResponse(BaseModel):
    """Response model for skin analysis"""
    disease_class: str
    confidence: float
    risk_level: str
    risk_score: float
    explanation: str
    disclaimer: str


class LocationRequest(BaseModel):
    """Request model for dermatologist search"""
    latitude: float
    longitude: float
    radius: Optional[int] = 5000  # meters


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SkinVision AI API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ml_model_loaded": ml_service.is_model_loaded()
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_skin_lesion(file: UploadFile = File(...)):
    """
    Analyze uploaded skin lesion image
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        AnalysisResponse with risk classification and confidence
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG)"
            )
        
        # Read and validate image
        image_data = await file.read()
        processed_image = image_service.preprocess_image(image_data)
        
        if processed_image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file"
            )
        
        # Perform ML inference
        prediction = ml_service.predict(processed_image)
        
        # Format response
        response = AnalysisResponse(
            disease_class=prediction["disease_class"],
            confidence=prediction["confidence"],
            risk_level=prediction["risk_level"],
            risk_score=prediction["risk_score"],
            explanation=prediction["explanation"],
            disclaimer=(
                "⚠️ This AI analysis is for screening purposes only and does not "
                "constitute a medical diagnosis. Always consult with a qualified "
                "dermatologist for professional medical advice."
            )
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/dermatologists/nearby")
async def find_nearby_dermatologists(location: LocationRequest):
    """
    Find nearby dermatologists using geolocation
    
    Args:
        location: LocationRequest with latitude, longitude, and radius
        
    Returns:
        List of nearby dermatologists with distance and contact info
    """
    try:
        # This would integrate with Google Maps Places API
        # For now, return mock data structure
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        if not api_key:
            # Return mock data for development
            return {
                "results": [
                    {
                        "name": "City Dermatology Clinic",
                        "address": "123 Medical Center Dr, City, State 12345",
                        "distance": 1.2,
                        "rating": 4.8,
                        "phone": "+1-555-0100",
                        "latitude": location.latitude + 0.01,
                        "longitude": location.longitude + 0.01
                    },
                    {
                        "name": "Skin Health Specialists",
                        "address": "456 Health Ave, City, State 12345",
                        "distance": 2.5,
                        "rating": 4.6,
                        "phone": "+1-555-0101",
                        "latitude": location.latitude - 0.015,
                        "longitude": location.longitude + 0.02
                    }
                ],
                "note": "Mock data - Configure GOOGLE_MAPS_API_KEY for real results"
            }
        
        # TODO: Implement actual Google Maps Places API integration
        # from services.maps_service import MapsService
        # maps_service = MapsService(api_key)
        # return maps_service.find_dermatologists(
        #     location.latitude,
        #     location.longitude,
        #     location.radius
        # )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find dermatologists: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

