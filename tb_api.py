from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from utils import preprocess_image, load_model_from_kaggle
import logging
from io import BytesIO
from PIL import Image
from similarity import check_similarity
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Class labels for predictions
CLASS_LABELS = {0: "Normal", 1: "Tuberculosis"}


# @router.on_event("startup")
# async def startup_event():
#     """Initialize model on startup with error handling"""
#     global tb_model
#     try:
#         logger.info("Starting TB model initialization...")
#         # Disable GPU and set memory growth
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#         tb_model = load_model_from_kaggle(
#             "khalednabawi",
#             'tb-chest-prediction',
#             "v1",
#             'tb_resnet.h5'
#         )
#         logger.info("TB model initialized successfully!")
#     except Exception as e:
#         logger.error(f"Error during startup: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Model initialization failed: {str(e)}"
#         )


@router.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "TB Detection API is running"
    }


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Endpoint to predict tuberculosis from uploaded chest X-ray
    Args:
        file: Uploaded image file
        request: FastAPI request object to access app state
    Returns:
        dict: Prediction results including class and confidence
    """
    try:
        logger.info(f"Receiving prediction request for file: {file.filename}")

        # Get model from app state
        tb_model = request.app.state.tb_model
        if tb_model is None:
            raise ValueError("TB model not initialized")

        # Validate image file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

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

        # Read and preprocess image
        img = Image.open(BytesIO(contents)).convert("RGB")

        img_array = preprocess_image(img)

        # Make prediction
        prediction = tb_model.predict(img_array, verbose=0)
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
