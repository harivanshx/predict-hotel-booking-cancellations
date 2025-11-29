"""
Configuration file for the hotel booking cancellation prediction pipeline.
Contains all parameters, paths, and settings used throughout the pipeline.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
EDA_PLOTS_DIR = LOGS_DIR / "eda_plots"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, EDA_PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "hotel_bookings.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.pkl"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_data.pkl"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_data.pkl"

# Model files
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest.pkl"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"

# Log files
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.log"
EDA_REPORT_FILE = LOGS_DIR / "eda_report.txt"
MODEL_EVALUATION_FILE = LOGS_DIR / "model_evaluation.txt"

# Data processing parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTLIER_METHOD = "IQR"  # Options: IQR, Z-score
OUTLIER_THRESHOLD = 1.5  # For IQR method
MISSING_VALUE_STRATEGY = {
    "numerical": "median",  # Options: mean, median, mode, drop
    "categorical": "mode"   # Options: mode, drop, constant
}

# Feature engineering parameters
LEAD_TIME_BINS = [0, 30, 90, 365, float('inf')]
LEAD_TIME_LABELS = ['short', 'medium', 'long', 'very_long']

# Columns to drop (if they exist and are not useful)
COLUMNS_TO_DROP = ['reservation_status', 'reservation_status_date']

# Target variable
TARGET_COLUMN = 'is_canceled'

# Features to encode
CATEGORICAL_FEATURES = [
    'hotel', 'meal', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type',
    'customer_type', 'arrival_date_month'
]

ORDINAL_FEATURES = []  # Add if any ordinal features exist

# Model parameters
MODELS_CONFIG = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
}

# Hyperparameter tuning parameters
TUNING_CONFIG = {
    'method': 'randomized',  # Options: grid, randomized
    'cv': 5,
    'n_iter': 20,  # For RandomizedSearchCV
    'scoring': 'roc_auc',
    'n_jobs': -1,
    'random_state': RANDOM_STATE
}

# XGBoost hyperparameter search space
XGBOOST_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Random Forest hyperparameter search space
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# SMOTE parameters for handling class imbalance
SMOTE_CONFIG = {
    'sampling_strategy': 'auto',
    'random_state': RANDOM_STATE,
    'k_neighbors': 5
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(PIPELINE_LOG_FILE),
            'formatter': 'standard',
            'level': 'INFO',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO',
    },
}
