from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from utils import preprocess_image
import logging
import json
from io import BytesIO
from PIL import Image
from similarity import check_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Class labels for predictions
CLASS_LABELS = {0: "Brain Tumor", 1: "Healthy"}


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Brain Tumor Detection API is running"
    }


# @router.on_event("startup")
# async def startup_event():
#     """Initialize model on startup"""
#     try:
#         logger.info("Starting brain model initialization...")
#         brain_model = load_model_from_kaggle(
#             "khalednabawi", 'brain-tumor-resnet', "v2", 'tb_resnet.h5')
#         logger.info("Brain model initialized successfully!")
#         # Store model in app state
#         router.app.state.brain_model = brain_model
#     except Exception as e:
#         logger.error(f"Error during startup: {str(e)}")
#         raise HTTPException(
#             status_code=500, detail="Model initialization failed")


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to predict brain tumor from uploaded image
    Args:
        file: Uploaded image file
        request: FastAPI request object to access app state
    Returns:
        dict: Prediction results including class and confidence
    """
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")

        # Get model from app state
        brain_model = request.app.state.brain_model
        if brain_model is None:
            raise ValueError("Brain tumor model not initialized")

        # Validate image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read and preprocess image
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
        img_array = preprocess_image(img)

        # Make prediction
        prediction = brain_model.predict(img_array, verbose=0)
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])

        result = CLASS_LABELS[predicted_class]
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
