# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor  # New estimator to handle NaNs
from sklearn.preprocessing import StandardScaler

def load_questionnaire_data():
    """Loads questionnaire data from CSV files."""
    questionnaire_files = [
        "patient_health_questionnaire_phq9.csv",
        "generalized_anxiety_disorder_scale_gad7.csv",
        "perceived_stress_scale_pss4.csv"
    ]
    dfs = []
    for file in questionnaire_files:
        df = pd.read_csv(os.path.join('data', file))
        dfs.append(df)
    questionnaire_data = pd.concat(dfs, ignore_index=True)
    return questionnaire_data

def load_dht_data():
    """Loads wearable device data from CSV files."""
    dht_files = [
        "garmin_epoch_run.csv",
        "garmin_epoch_walk.csv",
        "garmin_epoch_idle.csv",
        "oura_readiness.csv",
        "oura_extension_readiness.csv",
        "oura_sleep.csv",
        "oura_extension_sleep.csv",
        "oura_extension_activity.csv"
    ]
    dfs = []
    for file in dht_files:
        df = pd.read_csv(os.path.join('data', file))
        dfs.append(df)
    dht_data = pd.concat(dfs, ignore_index=True)
    return dht_data

def normalize_participant_ids(df):
    """Standardizes participant_id columns."""
    df['participant_id'] = df['participant_id'].astype(str).str.strip().str.upper()
    return df

def merge_datasets(dht_data, questionnaire_data):
    """Merges the wearable and questionnaire data."""
    merged_data = pd.merge(dht_data, questionnaire_data, on='participant_id', how='inner')
    return merged_data

def process_timestamps(merged_data):
    """Converts and processes timestamp columns."""
    # Rename and convert timestamps
    if 'summary_date' in merged_data.columns:
        merged_data.rename(columns={'summary_date': 'timestamp_dht'}, inplace=True)
    timestamp_cols = ['timestamp_dht', 'phq9_ts', 'gad_ts', 'pss4_ts']
    for col in timestamp_cols:
        if col in merged_data.columns:
            merged_data[col] = pd.to_datetime(merged_data[col], errors='coerce')
            merged_data[col] = merged_data[col].dt.floor('D')
    # Drop rows with missing timestamp_dht
    merged_data.dropna(subset=['timestamp_dht'], inplace=True)
    return merged_data

def clean_numeric_columns(merged_data):
    """Cleans and standardizes numeric columns."""
    numeric_cols = [
        'score_hrv_balance', 'rmssd', 'hr_lowest',
        'score_sleep_balance', 'score_total', 'deep',
        'light', 'rem', 'efficiency', 'onset_latency',
        'hr_average', 'phq9_total', 'gad_total', 'pss4_total'
    ]
    available_numeric_cols = [col for col in numeric_cols if col in merged_data.columns]
    merged_data[available_numeric_cols] = merged_data[available_numeric_cols].apply(pd.to_numeric, errors='coerce')
    return merged_data, available_numeric_cols

def optimize_data_types(merged_data, available_numeric_cols):
    """Optimizes data types to reduce memory usage."""
    # Downcast numeric columns
    for col in available_numeric_cols:
        if merged_data[col].dtype == 'float64':
            merged_data[col] = pd.to_numeric(merged_data[col], downcast='float')
        elif merged_data[col].dtype == 'int64':
            merged_data[col] = pd.to_numeric(merged_data[col], downcast='integer')
    # Convert object columns to category
    object_cols = merged_data.select_dtypes(include=['object']).columns
    for col in object_cols:
        merged_data[col] = merged_data[col].astype('category')
    return merged_data

def impute_missing_values(merged_data, available_numeric_cols):
    """Performs MICE imputation on missing values with scaling."""
    # Check for any NaN values before imputation
    print(f"NaN values before imputation: {merged_data[available_numeric_cols].isnull().sum().sum()}")

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(merged_data[available_numeric_cols])

    # Impute missing values using HistGradientBoostingRegressor
    mice_imputer = IterativeImputer(max_iter=50, random_state=42, tol=1e-3, verbose=2, 
                                    estimator=HistGradientBoostingRegressor())
    imputed_data = mice_imputer.fit_transform(scaled_data)

    # Inverse transform to original scale
    merged_data[available_numeric_cols] = scaler.inverse_transform(imputed_data)

    # Check for any NaN values after imputation
    print(f"NaN values after imputation: {merged_data[available_numeric_cols].isnull().sum().sum()}")

    return merged_data

def save_preprocessed_data(merged_data):
    """Saves the cleaned and imputed data to a compressed CSV."""
    if not os.path.exists('data'):
        os.makedirs('data')
    # Save as compressed CSV to reduce file size
    merged_data.to_csv('data/merged_data_preprocessed.csv.gz', index=False, compression='gzip')
    print("Preprocessed data saved to 'data/merged_data_preprocessed.csv.gz'.")

def main():
    # Load data
    questionnaire_data = load_questionnaire_data()
    dht_data = load_dht_data()

    # Normalize participant IDs
    questionnaire_data = normalize_participant_ids(questionnaire_data)
    dht_data = normalize_participant_ids(dht_data)

    # Merge datasets
    merged_data = merge_datasets(dht_data, questionnaire_data)
    print(f"Merged data shape: {merged_data.shape}")

    # Process timestamps
    merged_data = process_timestamps(merged_data)

    # Clean numeric columns
    merged_data, available_numeric_cols = clean_numeric_columns(merged_data)

    # Optimize data types
    merged_data = optimize_data_types(merged_data, available_numeric_cols)

    # Impute missing values
    merged_data = impute_missing_values(merged_data, available_numeric_cols)

    # Save preprocessed data
    save_preprocessed_data(merged_data)

if __name__ == "__main__":
    main()
