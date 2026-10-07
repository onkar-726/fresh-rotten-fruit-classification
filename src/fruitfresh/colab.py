"""Small helpers so the same notebooks run in Colab and on a laptop."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def in_colab() -> bool:
    return "google.colab" in sys.modules


def setup_workspace(mount_drive: bool = True) -> Path:
    """Folder where models, histories and predictions are stored.

    * FRUIT_WORK_DIR environment variable wins.
    * In Colab: Google Drive (survives runtime resets).
    * Elsewhere: ./workspace
    """
    env = os.getenv("FRUIT_WORK_DIR")
    if env:
        path = Path(env)
    elif in_colab():
        if mount_drive:
            from google.colab import drive

            drive.mount("/content/drive")
        path = Path("/content/drive/MyDrive/fruit_freshness_project")
    else:
        path = Path("workspace")
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpu_summary() -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return f"TensorFlow {tf.__version__} | GPUs: {[g.name for g in gpus] or 'none (CPU only)'}"
