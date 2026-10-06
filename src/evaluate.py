import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import IMAGE_SIZE, NUM_CLASSES


def build_baseline_cnn() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, 3)),

            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),

            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="baseline_cnn_time_block",
    )

    return model