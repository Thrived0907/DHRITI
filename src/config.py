"""Central configuration for the two-stage waste analysis pipeline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Model checkpoints (override with environment variables if you keep them elsewhere)
# ---------------------------------------------------------------------------
STAGE1_WEIGHTS = Path(os.getenv("STAGE1_WEIGHTS", MODELS_DIR / "stage1_yolov8.pt"))
STAGE2_WEIGHTS = Path(os.getenv("STAGE2_WEIGHTS", MODELS_DIR / "stage2_efficientnet_b0.pt"))

# ---------------------------------------------------------------------------
# Stage 1 - YOLOv8 detection defaults
# ---------------------------------------------------------------------------
DET_CONF_THRESHOLD = 0.25   # minimum box confidence
DET_IOU_THRESHOLD = 0.45    # NMS IoU threshold
DET_IMAGE_SIZE = 640
DET_MAX_OBJECTS = 20

# Context added around each box before handing the crop to Stage 2 (fraction of box size)
CROP_PADDING = 0.10
MIN_CROP_PIXELS = 16

# If YOLO finds nothing, classify the whole image with Stage 2 instead of returning empty
FULL_IMAGE_FALLBACK = True

# ---------------------------------------------------------------------------
# Stage 2 - EfficientNet-B0 defaults
# ---------------------------------------------------------------------------
CLS_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_COLOR = (124, 116, 104)  # neutral fill used when letterboxing a crop to a square

# RealWaste class folders (UCI RealWaste). Order = torchvision ImageFolder order (alphabetical).
# The trained checkpoint stores its own class list, which always takes precedence.
REALWASTE_CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

# ---------------------------------------------------------------------------
# Decision logic thresholds
# ---------------------------------------------------------------------------
STRONG_CONFIDENCE = 0.60    # a stage is "trusted" at or above this when the two stages disagree
MIN_FINAL_CONFIDENCE = 0.35  # below this the result is flagged for manual review
