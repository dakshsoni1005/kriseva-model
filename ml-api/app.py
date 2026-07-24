import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(
    title="Crop Recommendation Microservice API",
    description="Standalone ML API service for predicting optimal crops based on soil and climatic features.",
    version="1.0.0"
)

# Enable CORS for browser frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Payload Schema
class SoilParameters(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen content (kg/ha)", example=90)
    P: float = Field(..., ge=0, le=200, description="Phosphorus content (kg/ha)", example=42)
    K: float = Field(..., ge=0, le=300, description="Potassium content (kg/ha)", example=43)
    temperature: float = Field(..., ge=-10, le=60, description="Temperature (°C)", example=25.0)
    humidity: float = Field(..., ge=0, le=100, description="Relative Humidity (%)", example=80.0)
    ph: float = Field(..., ge=0, le=14, description="Soil pH level", example=6.5)
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall (mm)", example=200.0)

# Global variables for model & encoder
model = None
label_encoder = None

def load_model_artifacts():
    global model, label_encoder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'crop_model.pkl')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')

    if not os.path.exists(model_path):
        # Fallback to parent models folder if running from root
        model_path = os.path.join('models', 'crop_model.pkl')
        encoder_path = os.path.join('models', 'label_encoder.pkl')

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)
            print(f"Loaded ML model successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False

@app.on_event("startup")
def startup_event():
    load_model_artifacts()

@app.get("/")
def read_root():
    return {
        "service": "Crop Recommendation ML API",
        "status": "online",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "documentation": "GET /docs"
        }
    }

@app.get("/health")
def health_check():
    loaded = load_model_artifacts() if model is None else True
    return {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded
    }

@app.post("/predict")
def predict_crop(params: SoilParameters):
    global model, label_encoder
    if model is None or label_encoder is None:
        if not load_model_artifacts():
            raise HTTPException(status_code=531, detail="Model artifact missing. Please ensure crop_model.pkl is in models/")

    input_data = pd.DataFrame([{
        'N': params.N,
        'P': params.P,
        'K': params.K,
        'temperature': params.temperature,
        'humidity': params.humidity,
        'ph': params.ph,
        'rainfall': params.rainfall
    }])

    try:
        probabilities = model.predict_proba(input_data)[0]
        top_indices = np.argsort(probabilities)[::-1]

        # Primary prediction & confidence score
        top_class_raw = label_encoder.classes_[top_indices[0]]
        top_crop_name = str(top_class_raw).capitalize()
        confidence = int(round(probabilities[top_indices[0]] * 100))

        # Alternatives
        alternatives = []
        for idx in top_indices[1:3]:
            alt_raw = label_encoder.classes_[idx]
            alternatives.append(str(alt_raw).capitalize())

        return {
            "crop": top_crop_name,
            "confidence": confidence,
            "alternatives": alternatives
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
