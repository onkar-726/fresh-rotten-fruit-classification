from pathlib import Path
import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_environment() -> None:
    print("TensorFlow:", tf.__version__)
    print("NumPy:", np.__version__)

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("GPU available:", gpus)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("GPU available: None")


def verify_repo(repo_dir: str | Path) -> Path:
    repo_dir = Path(repo_dir)

    required_paths = [
        repo_dir / "data" / "time_block" / "train_time_block.csv",
        repo_dir / "data" / "time_block" / "validation_time_block.csv",
        repo_dir / "data" / "time_block" / "test_time_block.csv",
        repo_dir / "src" / "config.py",
        repo_dir / "src" / "data.py",
        repo_dir / "src" / "models.py",
        repo_dir / "src" / "evaluate.py",
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n"
            + "\n".join(missing)
        )

    return repo_dir