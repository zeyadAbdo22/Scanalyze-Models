# main.py
from fastapi import File, UploadFile, APIRouter, Request, HTTPException
from PIL import Image
import io
import logging
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from fastapi.responses import JSONResponse
from similarity import check_similarity

# from utils import preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Model classes and image preprocessing
CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


@router.get("/")
def read_root():
    return {"message": "X-ray Pneumonia Classifier is live!"}

# @app.on_event("startup")
# async def startup_event():
#     """Initialize model on startup"""
#     global model
#     try:
#         logger.info("Starting model initialization...")
#         model = load_model_from_kaggle()
#         logger.info("Model initialized successfully!")
#     except Exception as e:
#         logger.error(f"Error during startup: {str(e)}")
#         raise


def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize
    return img_array


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")
        contents = await file.read()

        # Parse similarity check response
        similar = check_similarity(contents)
        similarity_data = similar.body.decode()  # Convert bytes to string
        similarity_result = json.loads(similarity_data)  # Parse JSON string

        if similarity_result.get("prediction") == "not-medical":
            raise HTTPException(
                status_code=400,
                detail="File is not a valid medical image"
            )

        # Add logging for debugging
        logger.info(f"Similarity check result: {similarity_data}")

        img = Image.open(io.BytesIO(contents)).convert(
            "RGB")
        img_array = preprocess_image(img)

        pn_model = request.app.state.pneumonia_model

        if pn_model is None:
            raise ValueError("Model not initialized")

        # Make prediction
        prediction = pn_model.predict(img_array)
        # Convert sigmoid output to 0 or 1
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])  # Confidence score

        logger.info(f"Prediction complete: {CLASS_NAMES[predicted_class]}")
        return {
            "success": True,
            "filename": file.filename,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
