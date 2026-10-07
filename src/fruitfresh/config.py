"""Central configuration (Onkar's part: Baseline CNN + MobileNetV2).

Everything that is a "setting" lives here so notebooks and scripts stay short.
A few values can be overridden with environment variables (used for the quick
smoke test and for running the same code locally or in Colab).
"""
from __future__ import annotations

import os

PROJECT_NAME = "Fresh vs Rotten Fruit Classification"
KAGGLE_DATASET = "sriramr/fruits-fresh-and-rotten-for-classification"

# Class order is FIXED for the whole team (label index = position in list).
CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


# SMOKE=1 runs everything on a tiny synthetic dataset with 1 epoch per stage.
SMOKE = os.getenv("FRUIT_SMOKE", "0") == "1"

IMAGE_SIZE = (_int("FRUIT_IMAGE_SIZE", 224), _int("FRUIT_IMAGE_SIZE", 224))
BATCH_SIZE = _int("FRUIT_BATCH_SIZE", 32)
SEED = 42

# Official hold-out split (time-block split, see docs/TRAINING_GUIDE.md).
SPLIT_PARAMS = {
    "block_size": _int("FRUIT_BLOCK_SIZE", 25),
    "buffer_seconds": _int("FRUIT_BUFFER_SECONDS", 60),
    "val_fraction": 0.15,
    "test_fraction": 0.15,
}

# ---------------------------------------------------------------------------
# Models. Both take RAW RGB pixels (float 0..255); the right pre-processing is
# part of the model, so saved models are self-contained for your teammates.
# Any key below can be overridden per run (see MOBILENET_EXPERIMENTS).
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "baseline_cnn": {
        "title": "Baseline CNN (from scratch)",
        "kind": "scratch",
        "preprocess": "rescale_255",
        "epochs": 30,
        "lr": 1e-3,
        "patience": 6,
        "dropout": 0.3,
        "augment": True,         # random augmentation on training batches
        "class_weights": True,   # "balanced" class weights from the training split
        "image_size": None,      # None -> IMAGE_SIZE
    },
    "mobilenet_v2": {
        "title": "MobileNetV2 (transfer learning)",
        "kind": "transfer",
        "preprocess": "mobilenet_v2",
        "alpha": 1.0,            # MobileNetV2 width multiplier
        "head_epochs": 8,        # stage 1: backbone frozen, train the new head
        "head_lr": 1e-3,
        "ft_epochs": 12,         # stage 2: fine-tune the top backbone layers (0 = skip)
        "ft_lr": 2e-5,
        "ft_layers": 40,
        "patience": 4,
        "dropout": 0.3,
        "augment": True,
        "class_weights": True,
        "image_size": None,
    },
}

# ---------------------------------------------------------------------------
# MobileNetV2 experiments: ONE change per run relative to the default
# "mobilenet_v2" run above. Judged on the VALIDATION split only.
# ---------------------------------------------------------------------------
MOBILENET_EXPERIMENTS = {
    "head_only": {"ft_epochs": 0},                 # is fine-tuning worth it at all?
    "ft_top20": {"ft_layers": 20},                 # how deep to unfreeze
    "ft_top80": {"ft_layers": 80},
    "ft_lr_1e-5": {"ft_lr": 1e-5},                 # fine-tuning learning rate
    "ft_lr_5e-5": {"ft_lr": 5e-5},
    "no_class_weights": {"class_weights": False},
    "no_augmentation": {"augment": False},
    "dropout_0.5": {"dropout": 0.5},
    "alpha_0.75": {"alpha": 0.75},                 # a smaller MobileNetV2 (new stage 1)
    "size_160": {"image_size": (160, 160)},        # smaller input -> faster (new stage 1)
}

# Overrides that only change stage 2, so the default run's stage-1 model can be reused.
FINETUNE_ONLY_KEYS = {"ft_epochs", "ft_lr", "ft_layers"}
