"""tf.data input pipeline.

Images are decoded, resized and cached as uint8; they are converted to float
(0..255) at batch time. Models do their own pre-processing, so the pipeline is
the same for every model. Augmentation (training only) is applied per batch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import BATCH_SIZE, IMAGE_SIZE, SEED


def _load_image(path, label, image_size):
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, image_size)  # bilinear, float32
    return tf.cast(tf.round(img), tf.uint8), label


def build_augmenter(seed: int = SEED) -> keras.Sequential:
    """Random augmentation, applied to TRAINING batches only."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.08, fill_mode="reflect", seed=seed),
            layers.RandomZoom(0.15, fill_mode="reflect", seed=seed),
            layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed),
            layers.RandomBrightness(0.2, value_range=(0, 255), seed=seed),
            layers.RandomContrast(0.2, seed=seed),
        ],
        name="augmentation",
    )


def make_dataset(
    df: pd.DataFrame,
    dataset_root,
    image_size=IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    augment: bool = False,
    cache: bool = True,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Dataset of (float32 images 0..255, int labels) in the row order of ``df``
    (shuffled each epoch when ``training``)."""
    root = Path(dataset_root)
    paths = [str(root / p) for p in df["image_path"]]
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: _load_image(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels, num_classes: int | None = None) -> dict:
    """'Balanced' class weights: n_samples / (n_classes * count_c)."""
    labels = np.asarray(labels)
    num_classes = num_classes or int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}
