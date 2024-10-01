# src/data_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

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

def impute_missing_values(merged_data, available_numeric_cols):
    """Performs MICE imputation on missing values."""
    mice_imputer = IterativeImputer(max_iter=10, random_state=42)
    merged_data[available_numeric_cols] = mice_imputer.fit_transform(merged_data[available_numeric_cols])
    return merged_data

def save_preprocessed_data(merged_data):
    """Saves the cleaned and imputed data to CSV."""
    if not os.path.exists('data'):
        os.makedirs('data')
    merged_data.to_csv('data/merged_data_preprocessed.csv', index=False)
    print("Preprocessed data saved to 'data/merged_data_preprocessed.csv'.")

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

    # Impute missing values
    merged_data = impute_missing_values(merged_data, available_numeric_cols)

    # Save preprocessed data
    save_preprocessed_data(merged_data)

if __name__ == "__main__":
    main()
