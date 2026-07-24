import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Crop Knowledge Base Metadata
CROP_METADATA = {
    'rice': {
        'name': 'Rice',
        'category': 'Cereals & Grains',
        'season': 'Kharif (Monsoon)',
        'description': 'High water requirement staple crop thriving in clayey, alluvial soil with high nitrogen.',
        'optimal': {'N': 80, 'P': 48, 'K': 40, 'temp': 23.5, 'humidity': 85, 'ph': 6.5, 'rainfall': 240}
    },
    'maize': {
        'name': 'Maize (Corn)',
        'category': 'Cereals & Grains',
        'season': 'Kharif / Rabi',
        'description': 'Versatile cereal requiring well-drained fertile loam soils and moderate rainfall.',
        'optimal': {'N': 80, 'P': 48, 'K': 20, 'temp': 22.5, 'humidity': 65, 'ph': 6.2, 'rainfall': 85}
    },
    'chickpea': {
        'name': 'Chickpea',
        'category': 'Pulses & Legumes',
        'season': 'Rabi (Winter)',
        'description': 'Nitrogen-fixing pulse crop thriving in cooler climates and dry, well-drained soils.',
        'optimal': {'N': 35, 'P': 68, 'K': 80, 'temp': 19.5, 'humidity': 18, 'ph': 7.2, 'rainfall': 75}
    },
    'kidneybeans': {
        'name': 'Kidney Beans',
        'category': 'Pulses & Legumes',
        'season': 'Kharif / Rabi',
        'description': 'Protein-rich legume requiring rich soil, high phosphorus, and moderate temperatures.',
        'optimal': {'N': 28, 'P': 68, 'K': 20, 'temp': 20.0, 'humidity': 21, 'ph': 5.8, 'rainfall': 105}
    },
    'pigeonpeas': {
        'name': 'Pigeon Peas',
        'category': 'Pulses & Legumes',
        'season': 'Kharif',
        'description': 'Drought-tolerant leguminous crop suitable for semi-arid tropical regions.',
        'optimal': {'N': 28, 'P': 68, 'K': 20, 'temp': 32.5, 'humidity': 57, 'ph': 6.1, 'rainfall': 145}
    },
    'mothbeans': {
        'name': 'Moth Beans',
        'category': 'Pulses & Legumes',
        'season': 'Kharif',
        'description': 'Extremely drought-resistant legume ideal for arid sandy soils.',
        'optimal': {'N': 28, 'P': 48, 'K': 20, 'temp': 28.0, 'humidity': 52, 'ph': 6.7, 'rainfall': 52}
    },
    'mungbean': {
        'name': 'Mung Bean (Green Gram)',
        'category': 'Pulses & Legumes',
        'season': 'Kharif / Summer',
        'description': 'Short-duration pulse crop adaptable to warm temperatures and moderate humidity.',
        'optimal': {'N': 28, 'P': 48, 'K': 20, 'temp': 32.5, 'humidity': 85, 'ph': 6.7, 'rainfall': 48}
    },
    'blackgram': {
        'name': 'Black Gram',
        'category': 'Pulses & Legumes',
        'season': 'Kharif / Rabi',
        'description': 'Nutrient-rich pulse that improves soil fertility through biological nitrogen fixation.',
        'optimal': {'N': 48, 'P': 68, 'K': 20, 'temp': 30.0, 'humidity': 67, 'ph': 7.1, 'rainfall': 67}
    },
    'lentil': {
        'name': 'Lentil',
        'category': 'Pulses & Legumes',
        'season': 'Rabi (Winter)',
        'description': 'Cool-season legume requiring light to medium loamy soils.',
        'optimal': {'N': 28, 'P': 68, 'K': 20, 'temp': 24.0, 'humidity': 65, 'ph': 6.6, 'rainfall': 45}
    },
    'pomegranate': {
        'name': 'Pomegranate',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial',
        'description': 'Fruit crop favoring semi-arid conditions, high solar radiation, and well-drained soil.',
        'optimal': {'N': 28, 'P': 20, 'K': 40, 'temp': 21.5, 'humidity': 88, 'ph': 6.3, 'rainfall': 105}
    },
    'banana': {
        'name': 'Banana',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial',
        'description': 'High nutrient consumer requiring tropical warmth, rich nitrogen/phosphorus, and steady water.',
        'optimal': {'N': 100, 'P': 82, 'K': 50, 'temp': 28.0, 'humidity': 80, 'ph': 6.0, 'rainfall': 105}
    },
    'mango': {
        'name': 'Mango',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial (Summer Harvest)',
        'description': 'King of fruits thriving in warm climates with distinct dry weather during flowering.',
        'optimal': {'N': 28, 'P': 28, 'K': 30, 'temp': 31.5, 'humidity': 50, 'ph': 5.8, 'rainfall': 95}
    },
    'grapes': {
        'name': 'Grapes',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial',
        'description': 'Requires high potassium and phosphorus soil content, sunny days, and dry weather during ripening.',
        'optimal': {'N': 28, 'P': 132, 'K': 200, 'temp': 25.0, 'humidity': 82, 'ph': 6.0, 'rainfall': 70}
    },
    'watermelon': {
        'name': 'Watermelon',
        'category': 'Fruits & Horticulture',
        'season': 'Summer (Zaid)',
        'description': 'Warm-season vine requiring high nitrogen, warm temperatures, and sandy loam soil.',
        'optimal': {'N': 100, 'P': 18, 'K': 50, 'temp': 25.5, 'humidity': 85, 'ph': 6.5, 'rainfall': 50}
    },
    'muskmelon': {
        'name': 'Muskmelon',
        'category': 'Fruits & Horticulture',
        'season': 'Summer (Zaid)',
        'description': 'Requires warm dry climate, high sunlight, and well-drained sandy loam soil.',
        'optimal': {'N': 100, 'P': 18, 'K': 50, 'temp': 28.0, 'humidity': 92, 'ph': 6.3, 'rainfall': 25}
    },
    'apple': {
        'name': 'Apple',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial (Temperate)',
        'description': 'Temperate fruit requiring high potassium/phosphorus, cool temperatures, and abundant rain.',
        'optimal': {'N': 28, 'P': 132, 'K': 200, 'temp': 22.5, 'humidity': 92, 'ph': 6.0, 'rainfall': 112}
    },
    'orange': {
        'name': 'Orange',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial (Citrus)',
        'description': 'Citrus fruit thriving in subtropical climates with well-aerated sandy loam soil.',
        'optimal': {'N': 28, 'P': 18, 'K': 10, 'temp': 22.5, 'humidity': 92, 'ph': 6.7, 'rainfall': 110}
    },
    'papaya': {
        'name': 'Papaya',
        'category': 'Fruits & Horticulture',
        'season': 'Perennial (Tropical)',
        'description': 'Fast-growing tropical fruit requiring rich organic matter, balanced nutrients, and warm climate.',
        'optimal': {'N': 48, 'P': 58, 'K': 50, 'temp': 33.5, 'humidity': 92, 'ph': 6.8, 'rainfall': 145}
    },
    'coconut': {
        'name': 'Coconut',
        'category': 'Commercial / Plantation',
        'season': 'Perennial (Coastal)',
        'description': 'Coastal plantation palm requiring warm humid weather, sandy loam soil, and high rainfall.',
        'optimal': {'N': 28, 'P': 18, 'K': 30, 'temp': 26.5, 'humidity': 95, 'ph': 6.0, 'rainfall': 175}
    },
    'cotton': {
        'name': 'Cotton',
        'category': 'Commercial / Fibre',
        'season': 'Kharif',
        'description': 'Important cash crop favoring deep black cotton soil (regur), high nitrogen, and bright sunshine.',
        'optimal': {'N': 120, 'P': 48, 'K': 20, 'temp': 24.0, 'humidity': 80, 'ph': 7.0, 'rainfall': 80}
    },
    'jute': {
        'name': 'Jute (Golden Fibre)',
        'category': 'Commercial / Fibre',
        'season': 'Kharif',
        'description': 'High-humidity fibre crop requiring warm alluvial delta soils and heavy rainfall.',
        'optimal': {'N': 80, 'P': 48, 'K': 40, 'temp': 24.5, 'humidity': 75, 'ph': 6.7, 'rainfall': 175}
    },
    'coffee': {
        'name': 'Coffee',
        'category': 'Commercial / Plantation',
        'season': 'Perennial (Highland)',
        'description': 'Shade-loving plantation crop requiring high nitrogen, rich organic soil, and highland rain.',
        'optimal': {'N': 100, 'P': 28, 'K': 30, 'temp': 25.5, 'humidity': 60, 'ph': 6.6, 'rainfall': 155}
    }
}

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Crop Recommendation AI API",
    description="Machine Learning Powered Precision Agriculture Crop Recommendation System",
    version="1.0.0"
)

# Enable CORS for browser frontend fetch calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for Frontend UI
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

class SoilParameters(BaseModel):
    N: float = Field(..., ge=0, le=200, description="Nitrogen content in soil (kg/ha)", example=90)
    P: float = Field(..., ge=0, le=200, description="Phosphorus content in soil (kg/ha)", example=42)
    K: float = Field(..., ge=0, le=300, description="Potassium content in soil (kg/ha)", example=43)
    temperature: float = Field(..., ge=-10, le=60, description="Temperature in °C", example=20.8)
    humidity: float = Field(..., ge=0, le=100, description="Relative Humidity %", example=82.0)
    ph: float = Field(..., ge=0, le=14, description="Soil pH level", example=6.5)
    rainfall: float = Field(..., ge=0, le=500, description="Rainfall in mm", example=202.9)

# Global variables for model state
model = None
label_encoder = None

def load_artifacts():
    global model, label_encoder
    model_path = os.path.join('models', 'crop_recommendation_model.pkl')
    encoder_path = os.path.join('models', 'label_encoder.pkl')

    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)
            print("Successfully loaded ML model & label encoder.")
            return True
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            return False
    return False

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.get("/")
def read_root():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Crop Recommendation AI API is online. Access /docs for API schema."}

@app.get("/api/health")
def health_check():
    artifacts_loaded = load_artifacts() if model is None else True
    return {
        "status": "healthy",
        "model_loaded": artifacts_loaded,
        "supported_crops_count": len(CROP_METADATA)
    }

@app.get("/api/crops")
def get_crops():
    return CROP_METADATA

@app.post("/api/predict")
def predict_crop(params: SoilParameters):
    global model, label_encoder
    if model is None or label_encoder is None:
        if not load_artifacts():
            raise HTTPException(status_code=503, detail="ML Model not available. Please run training pipeline.")

    # Prepare input feature vector
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

        recommendations = []
        for idx in top_indices[:3]:
            crop_key = label_encoder.classes_[idx]
            prob = float(probabilities[idx])
            meta = CROP_METADATA.get(crop_key, {
                'name': crop_key.capitalize(),
                'category': 'Agriculture',
                'season': 'General',
                'description': 'Optimal crop match for input soil conditions.',
                'optimal': {}
            })
            recommendations.append({
                'crop_key': crop_key,
                'name': meta['name'],
                'confidence': round(prob * 100, 2),
                'category': meta['category'],
                'season': meta['season'],
                'description': meta['description'],
                'optimal': meta['optimal']
            })

        primary_recommendation = recommendations[0]

        return {
            "primary": primary_recommendation,
            "alternatives": recommendations[1:],
            "input_parameters": params.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict")
def predict_compact(params: SoilParameters):
    global model, label_encoder
    if model is None or label_encoder is None:
        if not load_artifacts():
            raise HTTPException(status_code=503, detail="ML Model not available.")

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

        top_crop_key = label_encoder.classes_[top_indices[0]]
        top_crop_name = CROP_METADATA.get(top_crop_key, {}).get('name', top_crop_key.capitalize())
        confidence = int(round(probabilities[top_indices[0]] * 100))

        alt_crops = []
        for idx in top_indices[1:3]:
            key = label_encoder.classes_[idx]
            alt_name = CROP_METADATA.get(key, {}).get('name', key.capitalize())
            alt_crops.append(alt_name)

        return {
            "crop": top_crop_name,
            "confidence": confidence,
            "alternatives": alt_crops
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

