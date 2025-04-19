import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image
from io import BytesIO
import json
import tensorflow as tf
import numpy as np
from utils import load_model_from_kaggle, preprocess_image
from similarity import check_similarity
from fastapi.responses import JSONResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define class labels
LUNG_CLASSES = {0: "Adenocarcinoma", 1: "Benign", 2: "Squamous Cell Carcinoma"}

# def preprocess_image(img: Image.Image):
#     img = img.resize((IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)


# def load_and_prepare_image(file: UploadFile):
#     """Reads image file and returns preprocessed image array."""
#     contents = file.file.read()
#     img = Image.open(io.BytesIO(contents)).convert("RGB")
#     return preprocess_image(img)


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Lung Cancer Detection API is running"}


# @app.post("/lung-cancer")
# async def predict_lung(file: UploadFile = File(...)):
#     try:
#         img_array = load_and_prepare_image(file)
#         label, confidence, raw = make_prediction(lung_model, img_array, LUNG_CLASSES)

#         return {
#             "success": True,
#             "prediction": label,
#             "confidence": round(confidence, 4),
#             "raw": raw
#         }
#     except Exception as e:
#         return {"success": False, "error": str(e)}


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
