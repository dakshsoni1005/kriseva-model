import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

def train_and_save():
    print("=" * 60)
    print(" CROP RECOMMENDATION AI - MODEL TRAINING ")
    print("=" * 60)

    # Path setup
    dataset_path = os.path.join('dataset', 'crop_recommendation.csv')
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load dataset & Clean data
    df = pd.read_csv(dataset_path)
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"Loaded dataset shape: {initial_shape} -> Cleaned shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target classes count: {df['label'].nunique()}")

    # Separate features and target
    X = df.drop(columns=['label'])
    y = df['label']

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    except ImportError:
        pass

    best_model = None
    best_accuracy = 0.0
    best_name = ""

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        print(f"Model: {name:<20} | Accuracy: {acc * 100:.2f}% | F1 Score: {f1:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_name = name

    print("-" * 60)
    print(f"Selected Best Model: {best_name} with Accuracy: {best_accuracy * 100:.2f}%")

    # Save artifacts
    model_path = os.path.join(models_dir, 'crop_recommendation_model.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    print("\nSaved artifacts:")
    print(f" - Model: {model_path}")
    print(f" - Encoder: {encoder_path}")
    print(f" - Scaler: {scaler_path}")
    print("=" * 60)

if __name__ == '__main__':
    train_and_save()
