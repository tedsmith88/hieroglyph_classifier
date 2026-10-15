"""Paths and defaults shared by the notebook, train.py and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Egyptian_hieroglyphs" / "Dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
WEIGHTS_DIR = ROOT / "weights"  # canonical final model (train.py); weights.pt is git-ignored, labels.csv is tracked
RUNS_DIR = ROOT / "runs"  # scratch outputs from the notebook, never the canonical model

NUM_CLASSES = 170
IMG_HW = (75, 50)  # (height, width) of every dataset image: 50 px wide, 75 px tall

BATCH_SIZE = 32
EPOCHS = 10
LR_FC = 1e-3
LR_LAYER4 = 1e-4
WEIGHT_DECAY = 0.0
SEED = 0

DEV_FRACTION = 0.2  # share of each training class held out as the dev set
DEV_SEED = 11
