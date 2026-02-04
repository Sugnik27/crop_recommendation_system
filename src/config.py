"""
This python file contains Basic configurations and paths of various files
"""

from pathlib import Path

# ROOT DIRECTORY

BASE_DIR: Path = Path(__file__).resolve().parents[1]


# DATA AND MODEL PATHS

DATA_DIR: Path = Path(BASE_DIR) / "data"
DATA_PATH: Path = Path(DATA_DIR) / "Crop_recommendation.csv"
CLEANED_DATA_PATH: Path = Path(DATA_DIR) / "cleaned_data.csv"


MODEL_DIR: Path = Path(BASE_DIR) / "models"
BEST_MODEL_PATH: Path = Path(MODEL_DIR) / "best_model.joblib"
FEATURE_PATH: Path = Path(MODEL_DIR) / "feature_columns.json"


TARGET_COLUMN: str = "label"



# TRAIN, TEST and CV

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
CV_FOLD: int = 5
N_JOBS: int = -1

# multi-class friendly metric
SCORING: str = "f1_macro"



print("Configuration loaded successfully.")