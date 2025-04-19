from fastapi import FastAPI, File, UploadFile, APIRouter, Request, HTTPException
import numpy as np
import json
import tensorflow as tf
from PIL import Image
from io import BytesIO
import logging
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from similarity import check_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


router = APIRouter()

# Global model variables
models = {
    "kidney": None,
    "knee": None
}

# Labels for each model
labels_map = {
    "kidney": ["Cyst", "Normal", "Stone", "Tumor"],
    "knee": ["Healthy", "Osteoporosis"]
}

# Load model from Kaggle Hub


# def load_model_from_kaggle(model_type: str):
#     try:
#         if model_type == "kidney":
#             model_path = kagglehub.model_download(
#                 "aliamrali/kidney-classification/keras/v1")
#             model_file = os.path.join(
#                 model_path, "resnet50_kidney_ct_augmented.h5")
#         elif model_type == "knee":
#             model_path = kagglehub.model_download(
#                 "aliamrali/knee-osteoporosis/keras/v1")
#             model_file = os.path.join(model_path, "Knee_Osteoporosis.h5")
#         else:
#             raise ValueError("Unknown model type")

#         if not os.path.exists(model_file):
#             raise FileNotFoundError(f"Model file not found: {model_file}")

#         return load_model(model_file, compile=False)
#     except Exception as e:
#         logger.error(f"Failed to load {model_type} model: {str(e)}")
#         raise

# # Startup event: Load both models


# @app.on_event("startup")
# async def startup_event():
#     logger.info("Loading models on startup...")
#     models["kidney"] = load_model_from_kaggle("kidney")
#     models["knee"] = load_model_from_kaggle("knee")
#     logger.info("Both models loaded successfully!")

# Root route


@router.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Medical Imaging API for Kidney and Knee Classification is running"
    }

# Helper function for preprocessing


def preprocess_image(img, model_type: str):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "kidney":
        return resnet_preprocess_input(img_array)
    elif model_type == "knee":
        return mobilenet_preprocess_input(img_array)
    else:
        raise ValueError("Invalid model type for preprocessing")

# Endpoint for Kidney Classification


@router.post("/kidney/predict")
async def predict_kidney(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Received kidney image: {file.filename}")
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

        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img, model_type="kidney")

        kidney_model = request.app.state.kidney_model

        if kidney_model is None:
            raise ValueError("Kidney model is not loaded")

        prediction = kidney_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return {
            "success": True,
            "filename": file.filename,
            "prediction": labels_map["kidney"][predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Kidney prediction error: {str(e)}")
        return {"success": False, "error": str(e)}

# Endpoint for Knee Osteoporosis Classification


@router.post("/knee/predict")
async def predict_knee(request: Request, file: UploadFile = File(...)):
    try:
        logger.info(f"Received knee image: {file.filename}")
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

        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = preprocess_image(img, model_type="knee")

        knee_model = request.app.state.knee_model

        if knee_model is None:
            raise ValueError("Kidney model is not loaded")

        prediction = knee_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        return {
            "success": True,
            "filename": file.filename,
            "prediction": labels_map["knee"][predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Knee prediction error: {str(e)}")
        return {"success": False, "error": str(e)}
