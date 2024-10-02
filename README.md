# AWARE: Analysis of Wearables for Assessing Risk and Evaluation

This repository contains the complete code and instructions to replicate the analysis of predicting mental health risk using wearable device data. The focus is on heart rate variability (HRV) metrics and mental health questionnaire data. The analysis includes data preprocessing with MICE imputation, data transformation, modeling using Convolutional Neural Networks (CNNs), and evaluation of the models.

**Note:** Due to the sensitivity of the data, the dataset files are not included in this repository. Researchers intending to replicate this analysis should use their own data or obtain the necessary datasets from appropriate sources.

---

## Repository Structure

```
AWARE-main/
├── data/
│   └── [Project data files here]
├── notebooks/
│   └── mental_health_risk_prediction.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── utils.py
│   └── __init__.py
├── models/
│   └── hrv_cnn_model.h5
├── README.md
├── requirements.txt
└── LICENSE
```

- **data/**: Directory to place the relevant dataset files.
- **notebooks/**: Jupyter notebooks for interactive analysis.
- **src/**: Source code modules for data preprocessing, model training, and utility functions.
- **models/**: Directory to save trained models.
- **README.md**: Detailed instructions and information about the project.
- **requirements.txt**: List of required Python packages.
- **LICENSE**: License information for the project.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab (optional, for running notebooks)
- Git (optional, for cloning the repository)

### Data Files

**Important:** The data files required for this analysis are not included due to sensitivity concerns. Obtain similar datasets to replicate the analysis.

#### Questionnaire Data Files

Place the questionnaire data files in the `data/` directory. The expected files are:

- `patient_health_questionnaire_phq9.csv`
- `generalized_anxiety_disorder_scale_gad7.csv`
- `perceived_stress_scale_pss4.csv`

These files should contain the mental health questionnaire responses for participants.

#### Wearable Device Data Files

Place the wearable device data files in the `data/` directory. The expected files are:

- `garmin_epoch_run.csv`
- `garmin_epoch_walk.csv`
- `garmin_epoch_idle.csv`
- `oura_readiness.csv`
- `oura_extension_readiness.csv`
- `oura_sleep.csv`
- `oura_extension_sleep.csv`
- `oura_extension_activity.csv`

These files should contain HRV metrics and other relevant data collected from wearable devices.

**Note:** Ensure that all data files are in CSV format and have the necessary columns as described below.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/roy-saurabh/AWARE-main.git
   cd AWARE-main
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file contains all the necessary packages such as pandas, numpy, scikit-learn, TensorFlow, Keras, matplotlib, seaborn, and others.

---

## Replicating the Analysis

### 1. Data Preprocessing

The preprocessing includes loading and merging datasets, handling missing values using MICE imputation, and data cleaning.

#### Run the Data Preprocessing Script

```bash
python src/data_preprocessing.py
```

This script performs the following steps:

- Loads the wearable device data and questionnaire data from the files placed in the `data/` directory.
- Normalizes `participant_id` columns and merges the datasets on `participant_id`.
- Converts timestamp columns to datetime objects.
- Cleans and standardizes numeric columns.
- Performs Multiple Imputation by Chained Equations (MICE) to handle missing values.
- Saves the preprocessed data for modeling.

#### `data_preprocessing.py`

**Note:** Replace the filenames in `data/` directory with the actual filenames of the datasets if they differ.

### 2. Modeling

The modeling involves transforming HRV time-series data into Gramian Angular Field (GAF) images and training a CNN.

#### Run the Model Training Script

```bash
python src/model_training.py
```

This script performs the following steps:

- Loads the preprocessed data (`merged_data_preprocessed.csv`).
- Transforms the `score_hrv_balance` time-series data into GAF images.
- Splits the data into training and testing sets.
- Defines and compiles the CNN architecture.
- Trains the model with early stopping and learning rate reduction callbacks.
- Evaluates the model on the test set and saves the trained model.

### 3. Utility Functions

Utility functions used across different scripts.

#### `utils.py`

### 4. Colab Notebook

An interactive Colab notebook for exploratory data analysis and visualization.

#### `mental_health_risk_prediction.ipynb`

Create the notebook using the scripts and code provided in `data_preprocessing.py` and `model_training.py`. The notebook would include:

- Data loading and exploration
- Visualization of data distributions
- Steps for data preprocessing and transformation
- Model training and evaluation
- Visualization of model performance metrics

### 5. Model File

#### `hrv_cnn_model.h5`

The `hrv_cnn_model.h5` file is the saved trained model generated after running `model_training.py`. Generate this file by running:

```bash
python src/model_training.py
```

---

## Dependencies

All required packages are listed in `requirements.txt`.

#### `requirements.txt`

```
pandas
numpy
scikit-learn
tensorflow
keras
matplotlib
seaborn
pyts
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Detailed Instructions

### Running the Analysis

1. **Ensure Data Availability**

   - Place the questionnaire data files (`patient_health_questionnaire_phq9.csv`, `generalized_anxiety_disorder_scale_gad7.csv`, `perceived_stress_scale_pss4.csv`) in the `data/` directory.
   - Place the wearable device data files (`garmin_epoch_run.csv`, `oura_readiness.csv`, etc.) in the `data/` directory.

2. **Data Preprocessing**

   Run the data preprocessing script:

   ```bash
   python src/data_preprocessing.py
   ```

   This will generate `merged_data_preprocessed.csv` in the `data/` directory.

3. **Model Training**

   Run the model training script:

   ```bash
   python src/model_training.py
   ```

   This will train the CNN model and save it as `hrv_cnn_model.h5` in the `models/` directory.

4. **Results and Evaluation**

   The evaluation metrics and plots will be displayed during the model training process. Explore the `mental_health_risk_prediction.ipynb` notebook for an interactive analysis.

### Reproducing the Results

To reproduce the results, follow the steps above. Use the same random seed (`random_state=42`) for data splitting and model training to obtain consistent results.

**Note:** Since the data files are not included due to sensitivity, use the relevant datasets. The scripts are designed to work with data files that have similar structures to those expected by the code.

### Customization

- **Adjusting Model Parameters**

  Modify the CNN architecture, hyperparameters, and training settings in the `model_training.py` script.

- **Using Different Data**

  Replace the data files in the `data/` directory with the datasets, ensuring they have the necessary columns.

---

## Repository Guidelines

- **Code Organization**

  The code is organized into modules for better readability and maintenance. Functions are documented with comments explaining their purpose.

- **Inline Comments**

  Important code sections include inline comments to explain the logic and any important considerations.

- **Readability**

  Variable names are descriptive to enhance understanding of the code flow.

---

## Contributing

To contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for the feature or bug fix.
3. Commit the changes with clear messages.
4. Push to the forked repository.
5. Submit a pull request detailing the changes.

---

## Contact

For any questions or issues, please open an issue on the repository or contact the maintainer at [roy.saurabh@pm.me](mailto:roy.saurabh@pm.me).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
