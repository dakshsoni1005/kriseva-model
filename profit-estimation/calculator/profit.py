import os
import pandas as pd

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def load_datasets(dataset_dir=DATASET_DIR):
    """
    Step 4: Load crop cost and market price datasets.
    """
    cost_path = os.path.join(dataset_dir, "crop_cost.csv")
    price_path = os.path.join(dataset_dir, "market_price.csv")
    
    if not os.path.exists(cost_path):
        raise FileNotFoundError(f"Crop cost dataset not found at {cost_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Market price dataset not found at {price_path}")

    cost_df = pd.read_csv(cost_path)
    price_df = pd.read_csv(price_path)
    
    return cost_df, price_df

def get_crop_cost(crop_name, cost_df):
    """
    Step 5: Find Crop Cost per Hectare
    """
    crop_match = cost_df[cost_df['Crop'].str.lower() == crop_name.lower()]
    if crop_match.empty:
        raise ValueError(f"Cost information for crop '{crop_name}' not found in dataset.")
    return float(crop_match.iloc[0]['CostPerHectare'])

def get_market_price(crop_name, price_df):
    """
    Step 6: Find Market Price per Quintal
    """
    crop_match = price_df[price_df['Crop'].str.lower() == crop_name.lower()]
    if crop_match.empty:
        raise ValueError(f"Market price for crop '{crop_name}' not found in dataset.")
    return float(crop_match.iloc[0]['PricePerQuintal'])

def estimate_profit(crop, yield_val, farm_area, district=None, recommended_crop=None, dataset_dir=DATASET_DIR):
    """
    Calculates profit based on farmer input.
    Steps 5 - 12:
    - Step 5: Find Crop Cost
    - Step 6: Find Market Price
    - Step 7: Calculate production (Yield * Area)
    - Step 8: Convert ton to Quintal (1 Ton = 10 Quintal)
    - Step 9: Gross income (Production in Quintals * Market Price)
    - Step 10: Total cultivation cost (CostPerHectare * Area)
    - Step 11: Net profit (Gross Income - Cultivation Cost)
    - Step 12: Profit report dictionary
    """
    cost_df, price_df = load_datasets(dataset_dir)
    
    # Standardize casing from dataset if found
    crop_match = price_df[price_df['Crop'].str.lower() == crop.lower()]
    canonical_crop_name = crop_match.iloc[0]['Crop'] if not crop_match.empty else crop

    cost_per_hectare = get_crop_cost(canonical_crop_name, cost_df)
    price_per_quintal = get_market_price(canonical_crop_name, price_df)
    
    # Step 7: Calculate production in Tons
    production_tons = float(yield_val) * float(farm_area)
    
    # Step 8: Convert ton to Quintal (1 Ton = 10 Quintals)
    production_quintals = production_tons * 10.0
    
    # Step 9: Gross income
    gross_income = production_quintals * price_per_quintal
    
    # Step 10: Total cultivation cost
    cultivation_cost = cost_per_hectare * float(farm_area)
    
    # Step 11: Net profit
    net_profit = gross_income - cultivation_cost
    
    # Step 12: Profit report structure
    report = {
        "crop": canonical_crop_name,
        "yield": round(float(yield_val), 2),
        "farmArea": round(float(farm_area), 2),
        "production": round(production_tons, 2),
        "grossIncome": round(gross_income, 2),
        "cultivationCost": round(cultivation_cost, 2),
        "netProfit": round(net_profit, 2)
    }

    if district:
        report["district"] = district
    if recommended_crop:
        report["recommendedCrop"] = recommended_crop
        
    return report

if __name__ == "__main__":
    # Test step 12 example
    sample_report = estimate_profit("Groundnut", 3.8, 2)
    print("Sample Profit Report:", sample_report)
