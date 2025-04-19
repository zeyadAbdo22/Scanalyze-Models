import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from io import BytesIO
import json
import tensorflow as tf
import numpy as np
from utils import preprocess_image
from similarity import check_similarity
from fastapi.responses import JSONResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define class labels
LUNG_CLASSES = {0: "Adenocarcinoma", 1: "Benign", 2: "Squamous Cell Carcinoma"}


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Lung Cancer Detection API is running"}


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to predict lung cancer from uploaded chest CT scan
    Args:
        file: Uploaded image file
        request: FastAPI request object to access app state
    Returns:
        dict: Prediction results including class and confidence
    """
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")

        # Get model from app state
        lung_model = request.app.state.lung_cancer_model
        if lung_model is None:
            raise ValueError("Lung cancer model not initialized")

        # Validate image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and preprocess image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img)

        # Make prediction
        prediction = lung_model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])

        result = LUNG_CLASSES[predicted_class]
        logger.info(f"Prediction complete: {result}")

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result,
            "confidence": confidence,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
