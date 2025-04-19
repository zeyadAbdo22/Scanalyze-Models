import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras import backend as K


def load_model_from_azure(model_url: str):
    """Load model from Azure Blob Storage with error handling."""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN backend

        # Download the model file from Azure Blob Storage
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            model_file_path = "/tmp/model.h5"
            with open(model_file_path, "wb") as model_file:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)
            model = load_model(model_file_path, compile=False)
            print(f"Model loaded successfully from {model_url}")
            return model
        else:
            raise Exception(f"Failed to download model from {model_url}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def preprocess_image(img):
    """Preprocesses the uploaded image for ResNet50."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Normalize using ResNet50's preprocessing
    img_array = preprocess_input(img_array)
    return img_array
