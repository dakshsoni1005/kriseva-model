import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    max_error
)

def evaluate_regression_model():
    """
    Dedicated evaluation script for Crop Yield Regression Model.
    Evaluates MAE, RMSE, R² Score, MAPE, and Max Error.
    """
    dataset_path = os.path.join('dataset', 'yield.csv')
    model_path = os.path.join('models', 'yield_model.pkl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training/train_yield.py first.")

    print("==========================================================")
    print("       CROP YIELD MODEL REGRESSION EVALUATION REPORT       ")
    print("==========================================================")
    
    # Load dataset & model
    df = pd.read_csv(dataset_path)
    pipeline = joblib.load(model_path)

    categorical_cols = ['Crop', 'District', 'Soil', 'Season', 'Irrigation']
    numerical_cols = ['Rainfall', 'Temperature', 'Humidity', 'N', 'P', 'K', 'pH']
    target_col = 'Yield'

    X = df[categorical_cols + numerical_cols]
    y = df[target_col]

    # Train / Test split matching train_yield.py
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # 1. MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_test, y_pred)

    # 2. RMSE (Root Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # 3. R² Score (Coefficient of Determination)
    r2 = r2_score(y_test, y_pred)

    # Additional Regression Insights
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    max_err = max_error(y_test, y_pred)

    print("\n--- REGRESSION EVALUATION METRICS TABLE ---")
    print("-------------------------------------------------------------------------------------")
    print(f"{'Metric':<25} | {'Formula / Concept':<35} | {'Calculated Value':<15}")
    print("-------------------------------------------------------------------------------------")
    print(f"{'MAE (Mean Absolute Error)':<25} | {'(1/n) Sum |y - y_hat|':<35} | {mae:.4f} Ton/Ha")
    print(f"{'RMSE (Root Mean Sq Error)':<25} | {'sqrt( (1/n) Sum (y - y_hat)^2 )':<35} | {rmse:.4f} Ton/Ha")
    print(f"{'R2 Score (Variance)':<25} | {'1 - (SS_res / SS_tot)':<35} | {r2:.4f} ({r2*100:.2f}%)")
    print(f"{'MAPE (Mean Abs % Error)':<25} | {'(1/n) Sum |(y - y_hat)/y| * 100':<35} | {mape:.2f}%")
    print(f"{'Max Absolute Error':<25} | {'max(|y - y_hat|)':<35} | {max_err:.4f} Ton/Ha")
    print("-------------------------------------------------------------------------------------\n")

    print("METRIC INTERPRETATION:")
    print(f" 1. MAE = {mae:.4f} Ton/Ha:")
    print(f"    Product predictions are off by an average of only {mae:.2f} Ton/Ha.")
    print(f" 2. RMSE = {rmse:.4f} Ton/Ha:")
    print(f"    Measures standard deviation of residuals with heavier penalties for large outliers.")
    print(f" 3. R2 Score = {r2:.4f} ({r2*100:.2f}%):")
    print(f"    The model captures {r2*100:.2f}% of the total variance in crop yield.")

    # Sample Predictions Comparison
    sample_df = pd.DataFrame({
        'Actual Yield (y)': y_test.values[:8],
        'Predicted Yield (y_hat)': np.round(y_pred[:8], 2),
        'Absolute Error |y - y_hat|': np.round(np.abs(y_test.values[:8] - y_pred[:8]), 2)
    })
    print("\n--- SAMPLE PREDICTIONS VS ACTUAL TEST VALUES ---")
    print(sample_df.to_string(index=False))

if __name__ == '__main__':
    evaluate_regression_model()
