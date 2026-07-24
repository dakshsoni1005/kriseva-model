# Crop Recommendation AI

An end-to-end Machine Learning powered Precision Agriculture system designed to recommend optimal crop choices based on soil composition (Nitrogen, Phosphorus, Potassium, pH) and environmental conditions (Temperature, Humidity, Annual Rainfall).

---

## 📁 Directory Structure

```
CropRecommendationAI/
├── dataset/
│   └── crop_recommendation.csv    # 2,200 sample dataset across 22 crops
├── models/
│   ├── crop_recommendation_model.pkl  # Trained Random Forest classifier
│   ├── label_encoder.pkl              # Target class encoder
│   └── scaler.pkl                     # Standard feature scaler
├── training/
│   ├── train_model.py             # Model training & model selection pipeline
│   └── evaluate.py                # Evaluation metrics & feature importances
├── api/
│   ├── main.py                    # FastAPI REST server & prediction router
│   └── static/
│       ├── index.html             # Modern glassmorphic Web UI
│       ├── style.css              # Custom styling & animations
│       └── script.js              # Interactivity, Chart.js radar & API client
├── notebooks/
│   └── crop_recommendation_eda.ipynb  # EDA, visualizations & model analysis
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🌿 Supported Crops (22 Categories)

- **Cereals & Grains:** Rice, Maize
- **Pulses & Legumes:** Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil
- **Fruits & Horticulture:** Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya
- **Commercial & Plantation:** Coconut, Cotton, Jute, Coffee

---

## 🚀 Quick Start & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model Pipeline
```bash
python training/train_model.py
```
This trains multiple classifiers (Random Forest, Extra Trees, Gradient Boosting) and exports the highest-performing model to `models/crop_recommendation_model.pkl`.

### 3. Evaluate Model Performance
```bash
python training/evaluate.py
```

### 4. Launch FastAPI Web Application
```bash
python -m uvicorn api.main:app --reload --port 8000
```
Open your browser and navigate to `http://localhost:8000` to interact with the visual interface and soil sliders.

---

## 📡 API Endpoints

- `GET /` — Serves the interactive Web Application frontend.
- `GET /api/health` — Returns API health status and model state.
- `GET /api/crops` — Returns metadata & optimal parameter ranges for all crops.
- `POST /api/predict` — Accepts soil metrics JSON payload and returns top recommended crops with probability scores.

#### Sample Request to `/api/predict`:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 23.5,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.0
}
```

#### Sample Response:
```json
{
  "primary": {
    "crop_key": "rice",
    "name": "Rice",
    "confidence": 98.4,
    "category": "Cereals & Grains",
    "season": "Kharif (Monsoon)",
    "description": "High water requirement staple crop thriving in clayey, alluvial soil with high nitrogen.",
    "optimal": {"N": 80, "P": 48, "K": 40, "temp": 23.5, "humidity": 85, "ph": 6.5, "rainfall": 240}
  },
  "alternatives": [ ... ]
}
```

---

## 📊 Exploratory Data Analysis (EDA)

Explore data distributions, correlation heatmaps, and feature importance charts in `notebooks/crop_recommendation_eda.ipynb`. Launch via Jupyter:

```bash
jupyter notebook notebooks/crop_recommendation_eda.ipynb
```
