import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Asset Directories
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Component Directories
PREPROCESSING_DIR = os.path.join(BASE_DIR, "preprocessing")
TRAINING_DIR      = os.path.join(BASE_DIR, "model_training")
FORECASTING_DIR   = os.path.join(BASE_DIR, "forecasting")

# Ensure asset directories exist (safety check)
for d in [DATA_DIR, MODEL_DIR, REPORTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# File Path Helpers
def get_data_path(filename):
    return os.path.join(DATA_DIR, filename)

def get_model_path(filename):
    return os.path.join(MODEL_DIR, filename)

def get_report_path(filename):
    return os.path.join(REPORTS_DIR, filename)
