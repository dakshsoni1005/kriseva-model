import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def train_xgboost_model():
    """
    Train dedicated XGBoost Regressor model on the 12-feature dataset.
    Features X: Crop, District, Soil, Season, Irrigation, Rainfall, Temperature, Humidity, N, P, K, pH
    Target y: Yield (Ton/Ha)
    """
    dataset_path = os.path.join('dataset', 'yield.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Run training/generate_dataset.py first.")

    print("==================================================")
    print("      XGBOOST REGRESSOR MODEL TRAINING            ")
    print("==================================================")
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    categorical_cols = ['Crop', 'District', 'Soil', 'Season', 'Irrigation']
    numerical_cols = ['Rainfall', 'Temperature', 'Humidity', 'N', 'P', 'K', 'pH']
    target_col = 'Yield'

    X = df[categorical_cols + numerical_cols]
    y = df[target_col]

    print(f"Dataset Shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical Features   ({len(numerical_cols)}): {numerical_cols}")
    print(f"Target Column (y):    '{target_col}' (Ton/Ha)\n")

    # Train / Test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )

    # Dedicated Tuned XGBoost Regressor
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])

    print("Training XGBoost Regressor Pipeline...")
    pipeline.fit(X_train, y_train)

    # Test set predictions & evaluation
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    # 5-Fold Cross Validation
    print("Performing 5-Fold Cross Validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)

    print("\n--------------------------------------------------")
    print("            XGBOOST EVALUATION METRICS            ")
    print("--------------------------------------------------")
    print(f"  R² Score (Test Set):     {r2:.4f} ({r2*100:.2f}%)")
    print(f"  5-Fold CV R² (Mean±Std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  RMSE:                    {rmse:.4f} Ton/Ha")
    print(f"  MAE:                     {mae:.4f} Ton/Ha")
    print(f"  MSE:                     {mse:.4f}")
    print(f"  Explained Variance:      {evs:.4f}")
    print("--------------------------------------------------")

    # Feature Importance Analysis
    fitted_preprocessor = pipeline.named_steps['preprocessor']
    fitted_model = pipeline.named_steps['model']
    
    cat_feature_names = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = list(cat_feature_names) + numerical_cols
    importances = fitted_model.feature_importances_

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n--- TOP 10 FEATURE IMPORTANCES ---")
    print(feat_imp_df.head(10).to_string(index=False))

    # Save Model Artifact
    os.makedirs('models', exist_ok=True)
    model_output_path = os.path.join('models', 'yield_model.pkl')
    joblib.dump(pipeline, model_output_path)
    print(f"\n[SUCCESS] Trained XGBoost Regressor model saved to: {model_output_path}")

if __name__ == '__main__':
    train_xgboost_model()
