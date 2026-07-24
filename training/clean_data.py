import os
import pandas as pd

def clean_yield_dataset():
    dataset_path = os.path.join('dataset', 'yield.csv')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    print("--- DATA CLEANING PIPELINE ---")
    df = pd.read_csv(dataset_path)
    initial_shape = df.shape
    print(f"Initial Dataset Shape: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # 1. Missing Values check & removal
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n1. Missing Values Found: {total_nulls}")
    if total_nulls > 0:
        df = df.dropna().reset_index(drop=True)
        print(f"   Removed missing rows. Remaining rows: {len(df)}")
    else:
        print("   No missing values detected.")

    # 2. Duplicate rows check & removal
    duplicates = df.duplicated().sum()
    print(f"\n2. Duplicate Rows Found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"   Removed duplicate rows. Remaining rows: {len(df)}")
    else:
        print("   No duplicate rows detected.")

    # 3. Categorical Columns cleaning (5 columns)
    categorical_cols = ['Crop', 'District', 'Soil', 'Season', 'Irrigation']
    print(f"\n3. Cleaning Categorical Columns ({len(categorical_cols)}): {categorical_cols}...")
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.title()

    # 4. Numerical Columns cleaning (7 features + 1 target)
    numerical_cols = ['Rainfall', 'Temperature', 'Humidity', 'N', 'P', 'K', 'pH', 'Yield']
    print(f"4. Cleaning Numerical Columns ({len(numerical_cols)}): {numerical_cols}...")
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna().reset_index(drop=True)
    
    # Validation filters
    df = df[(df['Rainfall'] > 0) & 
            (df['Temperature'] > -10) & 
            (df['Humidity'] >= 0) & (df['Humidity'] <= 100) &
            (df['N'] >= 0) & (df['P'] >= 0) & (df['K'] >= 0) &
            (df['pH'] >= 0) & (df['pH'] <= 14) &
            (df['Yield'] > 0)].reset_index(drop=True)

    # Unit formatting (Ton/Ha)
    df['Yield'] = df['Yield'].round(2)
    df['Rainfall'] = df['Rainfall'].round(1)
    df['Temperature'] = df['Temperature'].round(1)
    df['Humidity'] = df['Humidity'].round(1)
    df['N'] = df['N'].round(1)
    df['P'] = df['P'].round(1)
    df['K'] = df['K'].round(1)
    df['pH'] = df['pH'].round(2)

    final_shape = df.shape
    print(f"\nFinal Cleaned Dataset Shape: {final_shape[0]} rows, {final_shape[1]} columns")
    print(f"Cleaned records saved to {dataset_path}")

    df.to_csv(dataset_path, index=False)
    return df

if __name__ == '__main__':
    clean_yield_dataset()
