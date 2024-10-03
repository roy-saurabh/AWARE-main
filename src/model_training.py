# src/model_training.py

import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import plot_training_history, plot_confusion_matrix, plot_roc_curve
import os

def load_preprocessed_data():
    """Loads the preprocessed data from the compressed CSV."""
    data = pd.read_csv('data/merged_data_preprocessed.csv.gz', compression='gzip')
    return data

def transform_time_series_to_gaf(data):
    """Converts time-series data to GAF images."""
    participant_ids = data['participant_id'].unique()
    images = []
    labels = []
    for pid in participant_ids:
        participant_data = data[data['participant_id'] == pid].sort_values('timestamp_dht')
        time_series = participant_data['score_hrv_balance'].values
        # Handle cases with insufficient data
        if len(time_series) < 32:
            continue
        # Normalize time-series data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
        # Use the last 32 data points
        time_series_scaled = time_series_scaled[-32:]
        # Transform to GAF image
        transformer = GramianAngularField(image_size=32, method='summation')
        gaf = transformer.transform(time_series_scaled.reshape(1, -1))
        images.append(gaf.reshape(32, 32, 1))
        # Assuming binary labels based on 'phq9_total' threshold
        label = int(participant_data['phq9_total'].iloc[-1] >= 10)
        labels.append(label)
    X = np.array(images)
    y = np.array(labels)
    return X, y

def build_cnn_model(input_shape):
    """Defines the CNN architecture."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_model(X_train, y_train, X_val, y_val, input_shape):
    """Compiles and trains the CNN model with early stopping and learning rate reduction."""
    model = build_cnn_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model performance on test data."""
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy}, Loss: {test_loss}')
    
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_probs)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred_probs, y_pred

def save_model(model):
    """Saves the trained CNN model to disk."""
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/hrv_cnn_model.h5')
    print("Model saved to 'models/hrv_cnn_model.h5'.")

def main():
    """Main function for loading data, training the model, and evaluating results."""
    # Load preprocessed data
    data = load_preprocessed_data()
    
    # Transform time series data into GAF images
    X, y = transform_time_series_to_gaf(data)
    
    if len(X) == 0:
        print("No sufficient data to train the model. Exiting...")
        return
    
    # Split data into training and testing sets
    input_shape = (32, 32, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model, history = train_model(X_train, y_train, X_test, y_test, input_shape)
    
    # Save the trained model
    save_model(model)
    
    # Evaluate model performance
    y_pred_probs, y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()
