# Crop Yield Prediction Model & API Service

A Machine Learning pipeline and REST API server for predicting crop yields based on environmental factors, farm characteristics, and agricultural inputs.

## Project Structure

```text
yield-model/
│
├── dataset/
│   └── yield.csv             # Agricultural dataset with environmental & yield metrics
│
├── training/
│   ├── generate_dataset.py   # Script to generate realistic yield dataset
│   └── train_yield.py        # ML Pipeline training and evaluation script
│
├── models/
│   └── yield_model.pkl       # Serialized Random Forest model & preprocessing pipeline
│
├── api/
│   └── app.py                # Flask REST API server & Web Dashboard
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Features

- **Data Processing**: Preprocessing pipeline with `ColumnTransformer`, `OneHotEncoder` for categorical variables (`Crop`, `Season`), and `StandardScaler` for numerical variables (`Area_ha`, `Rainfall_mm`, `Temperature_C`, `Fertilizer_kg_per_ha`, `Pesticide_kg_per_ha`).
- **Machine Learning Model**: `RandomForestRegressor` achieving high predictive precision ($R^2 \approx 0.916$).
- **REST API**: Flask web service providing JSON prediction endpoints and health metrics.
- **Web UI Dashboard**: Modern glassmorphic web dashboard for testing crop yield predictions interactively.

## Getting Started

### 1. Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Generation & Model Training

To generate the dataset and train/save the model artifact (`models/yield_model.pkl`):

```bash
# Generate agricultural dataset
python training/generate_dataset.py

# Train ML model and save serialized pipeline
python training/train_yield.py
```

### 3. Launching the API & Dashboard

Run the Flask server:

```bash
python api/app.py
```

The app will start at `http://127.0.0.1:5000`.

## API Documentation

### POST `/predict`
Submit field parameters to receive the estimated yield per hectare and total estimated harvest.

#### Request Header:
`Content-Type: application/json`

#### Request Body Example:
```json
{
  "Crop": "Rice",
  "Season": "Kharif",
  "Area_ha": 10.0,
  "Rainfall_mm": 1200.0,
  "Temperature_C": 27.5,
  "Fertilizer_kg_per_ha": 150.0,
  "Pesticide_kg_per_ha": 4.5
}
```

#### Response Example:
```json
{
  "success": true,
  "predicted_yield_tons_per_ha": 4.1523,
  "total_estimated_production_tons": 41.523,
  "input": { ... }
}
```

### GET `/health`
Check server status and model loading state.
