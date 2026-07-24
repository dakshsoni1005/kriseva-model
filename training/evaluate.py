import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model():
    print("=" * 60)
    print(" CROP RECOMMENDATION AI - MODEL EVALUATION ")
    print("=" * 60)

    model_path = os.path.join('models', 'crop_recommendation_model.pkl')
    encoder_path = os.path.join('models', 'label_encoder.pkl')
    dataset_path = os.path.join('dataset', 'crop_recommendation.csv')

    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        print("Model artifacts missing. Run python training/train_model.py first.")
        return

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=['label'])
    y = df['label']
    y_true = label_encoder.transform(y)

    y_pred = model.predict(X)
    overall_acc = accuracy_score(y_true, y_pred)

    print(f"\nOverall Dataset Accuracy: {overall_acc * 100:.2f}%\n")
    print("--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    if hasattr(model, 'feature_importances_'):
        print("--- Feature Importances ---")
        feature_names = list(X.columns)
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        for rank, idx in enumerate(sorted_indices, 1):
            print(f"{rank}. {feature_names[idx]:<15}: {importances[idx] * 100:.2f}%")

    print("=" * 60)

if __name__ == '__main__':
    evaluate_model()
