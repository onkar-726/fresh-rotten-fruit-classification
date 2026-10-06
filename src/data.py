from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    TRAIN_CSV,
    VALIDATION_CSV,
    TEST_CSV,
)


LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def load_split(name: str) -> pd.DataFrame:
    paths = {
        "train": TRAIN_CSV,
        "validation": VALIDATION_CSV,
        "test": TEST_CSV,
    }

    if name not in paths:
        raise ValueError(
            f"Unknown split '{name}'. Use 'train', 'validation', or 'test'."
        )

    csv_path = Path(paths[name])

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "image_path",
        "class_name",
        "source_id",
        "source_filename",
        "capture_time",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def _resolve_image_path(
    image_path: str,
    dataset_root: str | Path,
) -> str:
    return str(Path(dataset_root) / image_path)


def preprocess_cnn_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_mobilenet_image(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image, seed=SEED)

    image = tf.image.random_brightness(
        image,
        max_delta=0.10,
        seed=SEED,
    )

    image = tf.image.random_contrast(
        image,
        lower=0.9,
        upper=1.1,
        seed=SEED,
    )

    return image


def build_dataset(
    df: pd.DataFrame,
    dataset_root: str | Path,
    training: bool = False,
    augment: bool = False,
    mobilenet: bool = False,
) -> tf.data.Dataset:

    paths = [
        _resolve_image_path(path, dataset_root)
        for path in df["image_path"].astype(str)
    ]

    labels = (
        df["class_name"]
        .map(LABEL_MAP)
        .astype(int)
        .to_numpy()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _load(path, label):
        if mobilenet:
            image = preprocess_mobilenet_image(path)
        else:
            image = preprocess_cnn_image(path)

        if training and augment:
            image = _augment_image(image)

        return image, tf.cast(label, tf.int32)

    dataset = dataset.map(
        _load,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def original_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly one original, non-augmented image per source photo.
    Prefer the row whose filename matches source_filename.
    """

    work = df.copy()

    work["_is_original"] = (
        work["image_path"]
        .map(lambda p: Path(str(p)).name)
        == work["source_filename"].astype(str)
    )

    work = work.sort_values(
        ["source_id", "_is_original", "capture_time"],
        ascending=[True, False, True],
    )

    result = (
        work
        .drop_duplicates("source_id", keep="first")
        .drop(columns="_is_original")
        .reset_index(drop=True)
    )

    return result


def compute_class_weights(
    train_df: pd.DataFrame,
) -> dict[int, float]:

    labels = (
        train_df["class_name"]
        .map(LABEL_MAP)
        .astype(int)
        .to_numpy()
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=labels,
    )

    return {
        class_id: float(weight)
        for class_id, weight in enumerate(weights)
    }