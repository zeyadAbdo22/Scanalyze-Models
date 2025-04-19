import logging
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from utils import load_model_from_azure
# Importing all api routers
from covid import router as covid_router
from pneumonia import router as pneumonia_router
from brain_api import router as brain_router
from tb_api import router as tb_router
from lung_cancer_api import router as lung_cancer_router
from kidnee import router as kidnee_router
from dr_api import router as dr_router

# Configure environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Disable tensorflow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI(title="Medical Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
   
@app.on_event("startup")
async def startup_event():
    """Initialize all models at startup"""
    try:
        logging.info("Loading all models at startup...")

        # Load Brain Tumor model from Azure
        app.state.brain_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet_brain_model.h5"
        )
        logging.info("Brain Tumor model loaded successfully")

        # Load TB model from Azure
        app.state.tb_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/tb_resnet.h5"
        )
        logging.info("TB model loaded successfully")

        # Load Lung Cancer model from Azure
        app.state.lung_cancer_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/lung-cancer-resnet-model.h5"
        )
        logging.info("Lung Cancer model loaded successfully")

        # Load Pneumonia model from Azure
        app.state.pneumonia_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/pneumonia_INCEP_classifier.h5"
        )
        logging.info("Pneumonia model loaded successfully")

        # Load Covid model from Azure
        app.state.covid_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Covid19_detection.h5"
        )
        logging.info("Covid model loaded successfully")

        # Load Kidney model from Azure
        app.state.kidney_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/resnet50_kidney_ct_augmented.h5"
        )
        logging.info("Kidney model loaded successfully")

        # Load Knee model from Azure
        app.state.knee_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Knee_Osteoporosis.h5"
        )
        logging.info("Knee model loaded successfully")

        # Load Diabetic Retinopathy model from Azure
        app.state.dr_model = load_model_from_azure(
            "https://scanalyzestorage.blob.core.windows.net/loadmodel/Diabetic-Retinopathy-ResNet50-model.h5"
        )
        logging.info("Diabetic-Retinopathy model loaded successfully")

    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Medical Scan Detection API is running",
    }

# Include the API routers with specific paths
app.include_router(brain_router, prefix="/Brain-Tumor", tags=["Brain Tumor Detection"])
app.include_router(tb_router, prefix="/Tuberculosis", tags=["TB Detection"])
app.include_router(lung_cancer_router, prefix="/Lung-Cancer", tags=["Lung Cancer Detection"])
app.include_router(covid_router, prefix="/Covid", tags=["Covid Detection"])
app.include_router(pneumonia_router, prefix="/Pneumonia", tags=["Pneumonia Detection"])
app.include_router(kidnee_router, prefix="/Kidnee", tags=["Kidney, Knee Detection"])
app.include_router(dr_router, prefix="/Diabetic-Retinopathy", tags=["Diabetic Retinopathy Detection"])
