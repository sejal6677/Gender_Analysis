# Gender Difference Analysis - IITB Track 1

This project analyzes gender differences in physiological and behavioral data using machine learning models with explainable AI techniques. The analysis focuses on predicting performance accuracy based on multimodal data including EEG, eye tracking, galvanic skin response, psychological metrics, and task-related variables.

## Description

The project investigates how physiological signals differ between male and female participants in a cognitive task setting. By employing separate models for each gender and using SHAP (SHapley Additive exPlanations) for feature importance analysis, the study provides insights into gender-specific patterns in physiological responses and their predictive power for task performance.

Key objectives:
- Preprocess and aggregate multimodal physiological data per participant
- Train gender-specific machine learning models (XGBoost and Random Forest)
- Analyze feature importance using SHAP values
- Compare model performance and gender-specific insights

## Features

- **Data Preprocessing**: Aggregates trial-level data to participant-level summaries with robust NaN handling
- **Gender-Specific Modeling**: Separate model training for male and female datasets
- **Explainable AI**: SHAP analysis for feature importance and model interpretability
- **Multiple Algorithms**: Comparison between XGBoost and Random Forest models
- **Comprehensive Evaluation**: RMSE, MAE, and MAPE metrics for model assessment

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Dependencies
Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib joblib
```

## Usage

The project is organized as a series of Jupyter notebooks. Run them in order:

1. **01_preprocessing_per_participant.ipynb**: Data preprocessing and aggregation
   - Loads raw CSV files (PSY.csv, EEG.csv, GSR.csv, EYE.csv, TIVA.csv)
   - Computes participant-level means with NaN imputation
   - Outputs processed dataset and NaN report

2. **02_gender_analysis_modeling.ipynb**: Model training and evaluation
   - Splits data by gender
   - Trains XGBoost models for male and female datasets
   - Evaluates model performance

3. **03_shap_analysis.ipynb**: SHAP feature importance analysis
   - Computes SHAP values for trained models
   - Generates summary plots and feature rankings

4. **04_report_and_findings.ipynb**: Results compilation and reporting

5. **05_experiment_documentation.ipynb**: Additional experiments and documentation
   - Includes Random Forest model comparison
   - Extended SHAP analysis

To run the notebooks:
```bash
jupyter notebook
```
Then open and execute the notebooks in numerical order.

## Project Structure

```
├── data/
│   ├── EEG.csv                                    # Electroencephalography data
│   ├── EYE.csv                                    # Eye tracking data
│   ├── GSR.csv                                    # Galvanic skin response data
│   ├── PSY.csv                                    # Psychological metrics
│   ├── TIVA.csv                                   # Task-related variables
│   ├── participant_summary_dataset.csv            # Processed full dataset
│   ├── participant_summary_male.csv               # Male subset
│   ├── participant_summary_female.csv             # Female subset
│   └── processed/
│       └── participant_summary_dataset.csv
├── models/
│   ├── model_female.pkl                           # XGBoost model for females
│   ├── model_male.pkl                             # XGBoost model for males
│   ├── rf_model_female.pkl                        # Random Forest model for females
│   ├── rf_model_male.pkl                          # Random Forest model for males
│   └── performance_predictor.pkl                  # General performance predictor
├── notebooks/
│   ├── 01_preprocessing_per_participant.ipynb
│   ├── 02_gender_analysis_modeling.ipynb
│   ├── 03_shap_analysis.ipynb
│   ├── 04_report_and_findings.ipynb
│   ├── 05_experiment_documentation.ipynb
│   └── data/
│       ├── shap_values_female.csv
│       └── shap_values_male.csv
└── README.md
```

## Results

### Model Performance
- **XGBoost**:
  - Male: RMSE = 0.0772, MAE = 0.0650, MAPE = 9.14%
  - Female: RMSE = 0.0854, MAE = 0.0701, MAPE = 10.17%
- **Random Forest**:
  - Male: RMSE = 0.0406, MAE = 0.0279, MAPE = 4.46%
  - Female: RMSE = 0.0395, MAE = 0.0301, MAPE = 4.36%

Random Forest outperformed XGBoost across all metrics for both genders.

### Key Insights
- **Male Dataset**: Strong emphasis on physiological signals (GSR, EEG) and behavioral features (anger, eye widen)
- **Female Dataset**: Highlighted micro-expressions (smirk, lip press) and EEG features
- Gender-specific patterns suggest complementary cognitive and behavioral strategies


