import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Locate model path relative to app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yield_model.pkl')

model_pipeline = None

def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Run training/train_yield.py first.")

# Load model upon starting app
load_model()

# Feature Metadata
FEATURE_META = {
    "categorical": ["Crop", "District", "Soil", "Season", "Irrigation"],
    "numerical": ["Rainfall", "Temperature", "Humidity", "N", "P", "K", "pH"],
    "target": "Yield",
    "crops": ["Rice", "Wheat", "Maize", "Cotton", "Soybean", "Barley"],
    "districts": ["Rajkot", "Ahmedabad", "Surat", "Junagadh", "Vadodara", "Amreli", "Bhavnagar"],
    "soils": ["Black", "Alluvial", "Red", "Sandy", "Clay"],
    "seasons": ["Kharif", "Rabi", "Summer"],
    "irrigations": ["Drip", "Canal", "Tube Well", "Rainfed", "Sprinkler"],
    "defaults": {
        "Crop": "Rice",
        "District": "Surat",
        "Soil": "Alluvial",
        "Season": "Kharif",
        "Irrigation": "Drip",
        "Rainfall": 1200.0,
        "Temperature": 27.5,
        "Humidity": 75.0,
        "N": 80.0,
        "P": 40.0,
        "K": 40.0,
        "pH": 6.8
    }
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crop Yield Prediction Engine (12 Features)</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b1329;
            --bg-card: rgba(22, 33, 62, 0.7);
            --border-card: rgba(255, 255, 255, 0.1);
            --accent-green: #10b981;
            --accent-emerald: #059669;
            --accent-cyan: #06b6d4;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --input-bg: rgba(15, 23, 42, 0.6);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at 15% 15%, #132448 0%, #0b1329 50%, #050b18 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-card);
            background: rgba(11, 19, 41, 0.8);
            backdrop-filter: blur(12px);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-cyan));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #ffffff, #a7f3d0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.4rem 0.9rem;
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--accent-green);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
        }

        main {
            max-width: 1250px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 1.5rem;
            flex: 1;
            display: grid;
            grid-template-columns: 1.3fr 0.7fr;
            gap: 2rem;
        }

        @media (max-width: 950px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .presets {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-card);
            color: var(--text-secondary);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .preset-btn:hover {
            background: rgba(16, 185, 129, 0.15);
            color: #ffffff;
            border-color: var(--accent-green);
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
        }

        @media (max-width: 700px) {
            .form-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 500px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        input, select {
            background: var(--input-bg);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent-green);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.25);
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-emerald));
            border: none;
            color: white;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(16, 185, 129, 0.45);
        }

        .results-panel {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .prediction-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.05));
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .yield-val {
            font-family: 'Outfit', sans-serif;
            font-size: 3.5rem;
            font-weight: 700;
            color: var(--accent-green);
            line-height: 1;
            margin: 0.5rem 0;
            text-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
        }

        .yield-unit {
            font-size: 1rem;
            color: var(--text-secondary);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }

        .metric-card {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-card);
            padding: 1rem;
            border-radius: 12px;
        }

        .metric-title {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        footer {
            text-align: center;
            padding: 1.5rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-card);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-icon">🌾</div>
            <h1>Yield Predict AI (12 Features)</h1>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>XGBoost Engine (R² = 92.56%)</span>
        </div>
    </header>

    <main>
        <div class="card">
            <div class="card-title">
                <span>Field Condition Parameters</span>
                <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: normal;">12 Input Features (X)</span>
            </div>

            <div class="presets">
                <span style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Presets:</span>
                <button class="preset-btn" onclick="applyPreset('Rice')">Rice (Surat)</button>
                <button class="preset-btn" onclick="applyPreset('Wheat')">Wheat (Rajkot)</button>
                <button class="preset-btn" onclick="applyPreset('Cotton')">Cotton (Junagadh)</button>
            </div>

            <form id="yieldForm" onsubmit="calculateYield(event)" class="form-grid">
                <div class="form-group">
                    <label for="Crop">Crop</label>
                    <select id="Crop" name="Crop" required>
                        {% for item in meta.crops %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label for="District">District</label>
                    <select id="District" name="District" required>
                        {% for item in meta.districts %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label for="Soil">Soil</label>
                    <select id="Soil" name="Soil" required>
                        {% for item in meta.soils %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label for="Season">Season</label>
                    <select id="Season" name="Season" required>
                        {% for item in meta.seasons %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label for="Irrigation">Irrigation</label>
                    <select id="Irrigation" name="Irrigation" required>
                        {% for item in meta.irrigations %}
                        <option value="{{ item }}">{{ item }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="form-group">
                    <label for="Rainfall">Rainfall (mm)</label>
                    <input type="number" step="1" id="Rainfall" name="Rainfall" value="1200" required>
                </div>

                <div class="form-group">
                    <label for="Temperature">Temperature (°C)</label>
                    <input type="number" step="0.1" id="Temperature" name="Temperature" value="27.5" required>
                </div>

                <div class="form-group">
                    <label for="Humidity">Humidity (%)</label>
                    <input type="number" step="0.1" id="Humidity" name="Humidity" value="75.0" required>
                </div>

                <div class="form-group">
                    <label for="N">Nitrogen (N)</label>
                    <input type="number" step="1" id="N" name="N" value="80" required>
                </div>

                <div class="form-group">
                    <label for="P">Phosphorus (P)</label>
                    <input type="number" step="1" id="P" name="P" value="40" required>
                </div>

                <div class="form-group">
                    <label for="K">Potassium (K)</label>
                    <input type="number" step="1" id="K" name="K" value="40" required>
                </div>

                <div class="form-group">
                    <label for="pH">Soil pH</label>
                    <input type="number" step="0.1" id="pH" name="pH" value="6.8" min="0" max="14" required>
                </div>

                <button type="submit" class="submit-btn">Predict Crop Yield</button>
            </form>
        </div>

        <div class="card results-panel">
            <div>
                <div class="card-title">Prediction Result (y)</div>
                
                <div class="prediction-box">
                    <div style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em;">Predicted Crop Yield</div>
                    <div class="yield-val" id="yieldDisplay">--</div>
                    <div class="yield-unit">Ton per Hectare (Ton/Ha)</div>
                </div>
            </div>

            <div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">Features Count</div>
                        <div class="metric-value">12 (5 Cat + 7 Num)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Model Accuracy</div>
                        <div class="metric-value">92.56% (R²)</div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Crop Yield Prediction Engine &bull; Feature Matrix X (12 Features) &rarr; Target y (Yield)
    </footer>

    <script>
        const presets = {
            'Rice': { Crop: 'Rice', District: 'Surat', Soil: 'Alluvial', Season: 'Kharif', Irrigation: 'Drip', Rainfall: 1400, Temperature: 28.0, Humidity: 80.0, N: 90, P: 45, K: 40, pH: 6.5 },
            'Wheat': { Crop: 'Wheat', District: 'Rajkot', Soil: 'Black', Season: 'Rabi', Irrigation: 'Canal', Rainfall: 650, Temperature: 20.0, Humidity: 60.0, N: 70, P: 35, K: 35, pH: 7.0 },
            'Cotton': { Crop: 'Cotton', District: 'Junagadh', Soil: 'Black', Season: 'Kharif', Irrigation: 'Sprinkler', Rainfall: 1100, Temperature: 30.0, Humidity: 70.0, N: 85, P: 40, K: 45, pH: 7.2 }
        };

        function applyPreset(key) {
            const p = presets[key];
            if (!p) return;
            for (const [k, v] of Object.entries(p)) {
                const el = document.getElementById(k);
                if (el) el.value = v;
            }
            calculateYield();
        }

        async function calculateYield(e) {
            if (e) e.preventDefault();
            
            const form = document.getElementById('yieldForm');
            const formData = new FormData(form);
            const data = {};
            const categoricalKeys = ['Crop', 'District', 'Soil', 'Season', 'Irrigation'];
            
            formData.forEach((val, key) => {
                data[key] = categoricalKeys.includes(key) ? val : parseFloat(val);
            });

            const yieldDisplay = document.getElementById('yieldDisplay');
            yieldDisplay.innerText = "...";

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                if (result.success) {
                    yieldDisplay.innerText = result.predicted_yield_ton_per_ha.toFixed(2);
                } else {
                    yieldDisplay.innerText = "Error";
                    alert(result.error || "Failed to make prediction");
                }
            } catch (err) {
                console.error(err);
                yieldDisplay.innerText = "Error";
            }
        }

        window.addEventListener('DOMContentLoaded', () => {
            calculateYield();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render interactive frontend web UI dashboard"""
    return render_template_string(
        HTML_TEMPLATE,
        meta=FEATURE_META
    )

@app.route('/health', methods=['GET'])
def health():
    """API health check endpoint"""
    return jsonify({
        "status": "online",
        "model_loaded": model_pipeline is not None,
        "model_path": MODEL_PATH,
        "feature_meta": FEATURE_META
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON API endpoint for crop yield prediction.
    Expects 12 Features in payload:
    Crop, District, Soil, Rainfall, Temperature, Humidity, N, P, K, pH, Season, Irrigation
    Returns:
    predicted_yield_ton_per_ha (Yield target y)
    """
    if model_pipeline is None:
        return jsonify({
            "success": False,
            "error": "Model is not loaded. Ensure models/yield_model.pkl exists."
        }), 500

    try:
        data = request.get_json(force=True)
        
        required_fields = ['Crop', 'District', 'Soil', 'Rainfall', 'Temperature', 'Humidity', 'N', 'P', 'K', 'pH', 'Season', 'Irrigation']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing required fields in payload: {missing}"
            }), 400

        input_df = pd.DataFrame([{
            'Crop': str(data['Crop']),
            'District': str(data['District']),
            'Soil': str(data['Soil']),
            'Rainfall': float(data['Rainfall']),
            'Temperature': float(data['Temperature']),
            'Humidity': float(data['Humidity']),
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'pH': float(data['pH']),
            'Season': str(data['Season']),
            'Irrigation': str(data['Irrigation'])
        }])

        predicted_yield = float(model_pipeline.predict(input_df)[0])

        return jsonify({
            "success": True,
            "predicted_yield_ton_per_ha": round(predicted_yield, 4),
            "unit": "Ton/Ha",
            "input": data
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
