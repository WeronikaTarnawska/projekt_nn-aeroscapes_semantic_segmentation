"""Aeroscapes-specific constants derived from EDA (notebooks/eda.ipynb).

Pulled out so configs reference values by name instead of hardcoding magic
numbers, and so all configs share one source of truth.
"""

# 11 foreground classes + background (index 0) — EDA confirmed 12 unique values.
NUM_CLASSES = 12
CLASS_NAMES = [
    "background",  # 0
    "person",  # 1
    "bike",  # 2
    "car",  # 3
    "drone",  # 4
    "boat",  # 5
    "animal",  # 6
    "obstacle",  # 7
    "construction",  # 8
    "vegetation",  # 9
    "road",  # 10
    "sky",  # 11
]

# Source images are 1280x720 (W x H, all identical per EDA).
SOURCE_HEIGHT = 720
SOURCE_WIDTH = 1280

# Background is ~23% of pixels but carries no semantic information. Configs can
# pass this as ignore_index to exclude it from loss & metrics.
IGNORE_INDEX_BACKGROUND = 0

# Pretrained TorchVision backbones expect ImageNet normalization. EDA showed
# Aeroscapes channel stats are close enough to use these directly.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Median-frequency-balanced class weights (Eigen & Fergus 2015), computed in
# notebooks/eda.ipynb on the official train split. Order matches CLASS_NAMES.
# Used as `weight=` for CrossEntropyLoss to compensate for the heavy pixel-share
# imbalance (vegetation/road/background hold ~90% of pixels; boat/animal/drone
# hold <0.1% combined). Median-frequency is gentler than pure inverse-frequency
# and avoids assigning multi-hundred weights to the rarest classes.
MEDIAN_FREQ_WEIGHTS = (
    0.02,  # background
    1.20,  # person
    8.11,  # bike
    1.50,  # car
    19.06,  # drone
    22.57,  # boat
    7.47,  # animal
    0.86,  # obstacle
    0.14,  # construction
    0.01,  # vegetation
    0.02,  # road
    0.11,  # sky
)
