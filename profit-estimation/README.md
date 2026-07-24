# Profit Estimation Module

A Python & Flask microservice for estimating crop production, gross income, cultivation costs, and net profit based on farm input data and market datasets.

---

## 📁 Directory Structure

```
profit-estimation/
├── dataset/
│   ├── crop_cost.csv       # Cultivation cost per hectare by crop
│   └── market_price.csv    # Market price per quintal by crop
├── models/                  # Placeholder for machine learning models
├── calculator/
│   └── profit.py           # Core calculation logic and data loader
├── api/
│   └── app.py              # Flask API service with /profit endpoint
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 📊 Dataset Specifications

### 1. `market_price.csv`
| Crop | PricePerQuintal |
| :--- | :--- |
| Groundnut | 6800 |
| Cotton | 7600 |
| Wheat | 2500 |
| Rice | 3000 |
| Castor | 6400 |

### 2. `crop_cost.csv`
| Crop | CostPerHectare |
| :--- | :--- |
| Groundnut | 55000 |
| Cotton | 60000 |
| Wheat | 30000 |
| Rice | 35000 |
| Castor | 40000 |

---

## 🧮 Step-by-Step Calculation Logic

1. **Farmer Input**: `crop`, `yield` (tons/hectare), `farmArea` (hectares).
2. **Dataset Lookup**:
   - `CostPerHectare` from `crop_cost.csv`
   - `PricePerQuintal` from `market_price.csv`
3. **Production Calculation**:
   $$\text{Production (Tons)} = \text{Yield} \times \text{Farm Area}$$
4. **Unit Conversion**:
   $$1\text{ Ton} = 10\text{ Quintals}$$
   $$\text{Production (Quintals)} = \text{Production (Tons)} \times 10$$
5. **Gross Income**:
   $$\text{Gross Income} = \text{Production (Quintals)} \times \text{PricePerQuintal}$$
6. **Cultivation Cost**:
   $$\text{Total Cultivation Cost} = \text{CostPerHectare} \times \text{Farm Area}$$
7. **Net Profit**:
   $$\text{Net Profit} = \text{Gross Income} - \text{Total Cultivation Cost}$$

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running the API
```bash
python api/app.py
```
The server will start at `http://localhost:5000`.

---

## 📡 API Endpoint

### `POST /profit`

#### Request Body:
```json
{
  "crop": "Groundnut",
  "yield": 3.8,
  "farmArea": 2
}
```

#### Response (200 OK):
```json
{
  "crop": "Groundnut",
  "yield": 3.8,
  "farmArea": 2.0,
  "production": 7.6,
  "grossIncome": 516800.0,
  "cultivationCost": 110000.0,
  "netProfit": 406800.0
}
```
