import os
import numpy as np
import pandas as pd

def generate_yield_dataset(num_samples=1500, seed=42):
    np.random.seed(seed)
    
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Soybean', 'Barley']
    districts = ['Rajkot', 'Ahmedabad', 'Surat', 'Junagadh', 'Vadodara', 'Amreli', 'Bhavnagar']
    soils = ['Black', 'Alluvial', 'Red', 'Sandy', 'Clay']
    seasons = ['Kharif', 'Rabi', 'Summer']
    irrigations = ['Drip', 'Canal', 'Tube Well', 'Rainfed', 'Sprinkler']
    
    crop_base_yield = {
        'Rice': 3.8,
        'Wheat': 3.5,
        'Maize': 4.2,
        'Cotton': 2.1,
        'Soybean': 2.5,
        'Barley': 3.0
    }
    
    crop_season_map = {
        'Rice': ['Kharif', 'Summer'],
        'Wheat': ['Rabi'],
        'Maize': ['Kharif', 'Rabi'],
        'Cotton': ['Kharif'],
        'Soybean': ['Kharif'],
        'Barley': ['Rabi']
    }
    
    soil_boost = {
        'Black': 0.35,
        'Alluvial': 0.45,
        'Clay': 0.20,
        'Red': 0.10,
        'Sandy': -0.25
    }
    
    irrigation_boost = {
        'Drip': 0.50,
        'Canal': 0.35,
        'Sprinkler': 0.30,
        'Tube Well': 0.25,
        'Rainfed': 0.0
    }
    
    data = []
    
    for _ in range(num_samples):
        crop = np.random.choice(crops)
        district = np.random.choice(districts)
        soil = np.random.choice(soils)
        season = np.random.choice(crop_season_map[crop])
        irrigation = np.random.choice(irrigations)
        
        # Environmental and Soil Chemical Properties (N, P, K, pH, Humidity, Temperature, Rainfall)
        if crop in ['Rice', 'Cotton']:
            rainfall = round(float(np.random.uniform(800, 2200)), 1)
            temp = round(float(np.random.uniform(22.0, 36.0)), 1)
            humidity = round(float(np.random.uniform(65.0, 95.0)), 1)
        elif crop in ['Wheat', 'Barley']:
            rainfall = round(float(np.random.uniform(350, 900)), 1)
            temp = round(float(np.random.uniform(12.0, 26.0)), 1)
            humidity = round(float(np.random.uniform(40.0, 75.0)), 1)
        else: # Maize, Soybean
            rainfall = round(float(np.random.uniform(500, 1400)), 1)
            temp = round(float(np.random.uniform(18.0, 32.0)), 1)
            humidity = round(float(np.random.uniform(50.0, 85.0)), 1)
            
        N = round(float(np.random.uniform(20.0, 140.0)), 1)
        P = round(float(np.random.uniform(10.0, 90.0)), 1)
        K = round(float(np.random.uniform(15.0, 120.0)), 1)
        pH = round(float(np.random.uniform(5.5, 8.5)), 2)
        
        # Agronomic yield calculation formula
        base = crop_base_yield[crop]
        s_effect = soil_boost[soil]
        i_effect = irrigation_boost[irrigation]
        
        # NPK nutrient effect
        npk_effect = 0.008 * N + 0.006 * P + 0.005 * K - 0.00003 * (N**2 + P**2 + K**2)
        
        # Optimal pH effect (around 6.5 - 7.2)
        ph_effect = -0.15 * ((pH - 6.8)**2) + 0.2
        
        # Rainfall and temperature effect
        opt_rain = 1200 if crop in ['Rice', 'Cotton'] else 650
        rain_effect = -0.000002 * ((rainfall - opt_rain) ** 2) + 0.3
        opt_temp = 28.0 if crop in ['Rice', 'Cotton', 'Maize', 'Soybean'] else 20.0
        temp_effect = -0.015 * ((temp - opt_temp) ** 2) + 0.2
        
        noise = np.random.normal(0, 0.18)
        
        yield_val = max(0.5, round(float(base + s_effect + i_effect + npk_effect + ph_effect + rain_effect + temp_effect + noise), 2))
        
        data.append({
            'Crop': crop,
            'District': district,
            'Soil': soil,
            'Rainfall': rainfall,
            'Temperature': temp,
            'Humidity': humidity,
            'N': N,
            'P': P,
            'K': K,
            'pH': pH,
            'Season': season,
            'Irrigation': irrigation,
            'Yield': yield_val
        })
        
    df = pd.DataFrame(data)
    os.makedirs('dataset', exist_ok=True)
    dataset_path = os.path.join('dataset', 'yield.csv')
    df.to_csv(dataset_path, index=False)
    print(f"Updated dataset generated with 12 features and 'Yield' target at {dataset_path}")

if __name__ == '__main__':
    generate_yield_dataset()
