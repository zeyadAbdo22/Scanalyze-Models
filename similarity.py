from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from fastapi import APIRouter
from utils import load_model_from_kaggle

model = load_model_from_kaggle(
    'aliamrali', 'medical_scan_checker', 'v2', 'medical_scan_checker.h5'
)


def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')

    if img.size != (224, 224):
        img = img.resize((224, 224))

    img = np.array(img)
    img = img / 255.0  # Normalization

    img = np.expand_dims(img, axis=0)
    return img


def check_similarity(file):

    img = prepare_image(file)

    prediction = model.predict(img)
    # predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)

    result = "medical" if prediction[0] > 0.5 else "not-medical"

    if result == "medical" and confidence >= 0.95:
        return JSONResponse(content={
            "prediction": result,
            "confidence": float(confidence)
        })
    else:
        return JSONResponse(content={
            "prediction": result,
            "confidence": float(confidence)
        })
