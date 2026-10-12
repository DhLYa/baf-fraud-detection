"""Paths, the train/validation split and evaluation settings shared by every notebook."""
from pathlib import Path

# src/baf/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Base.csv"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODELS_DIR = PROJECT_ROOT / "models"
SHAP_DIR = PROJECT_ROOT / "results" / "shap"

TARGET = "fraud_bool"

# Time-based split: fraud rate drifts across months, so later months are held out
TRAIN_MONTHS = [0, 1, 2, 3, 4, 5, 6]
VAL_MONTHS = [7]

# Decision threshold is set so this share of non-fraud validation rows is flagged (BAF benchmark convention)
TARGET_FPR = 0.05
AGE_GROUP_CUTOFF = 50

# customer_age is binned by decade. The 90s bucket is too small for a meaningful FPR (3 validation rows),
# so it is merged into 80+
AGE_BUCKET_CAP = 80
AGE_BUCKET_LABELS = {}
for decade in range(10, AGE_BUCKET_CAP + 1, 10):
    if decade < AGE_BUCKET_CAP:
        AGE_BUCKET_LABELS[decade] = f"{decade}-{decade + 9}"
    else:
        AGE_BUCKET_LABELS[decade] = f"{decade}+"
