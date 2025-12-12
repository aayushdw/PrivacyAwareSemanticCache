"""
Constants and configuration paths for the model evaluation pipeline.
"""
import os

# Get the absolute path to the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SMALL_TRAIN_PATH = os.path.join(DATA_DIR, "small", "questions_train.csv")
MEDIUM_TRAIN_PATH = os.path.join(DATA_DIR, "medium", "questions_train.csv")
LARGE_TRAIN_PATH = os.path.join(DATA_DIR, "large", "questions_train.csv")