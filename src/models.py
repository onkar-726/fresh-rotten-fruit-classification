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


def build_mobilenetv2() -> tuple[
    tf.keras.Model,
    tf.keras.Model,
]:
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        name="mobilenetv2_backbone",
    )

    backbone.trainable = False

    inputs = layers.Input(
        shape=(*IMAGE_SIZE, 3)
    )

    x = backbone(
        inputs,
        training=False,
    )

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="mobilenetv2_time_block",
    )

    return model, backbone


def set_fine_tune(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    n_layers: int = 30,
) -> tuple[int, int]:
    """
    Unfreeze the last n_layers of the MobileNetV2 backbone,
    while keeping all BatchNormalization layers frozen.
    """

    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    selected_layers = (
        backbone.layers[-n_layers:]
        if n_layers > 0
        else []
    )

    for layer in selected_layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    # Always keep BatchNorm frozen.
    for layer in backbone.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False

    trainable_layers = [
        layer
        for layer in backbone.layers
        if layer.trainable
    ]

    trainable_weight_layers = [
        layer
        for layer in trainable_layers
        if layer.weights
    ]

    batchnorm_trainable = [
        layer.name
        for layer in backbone.layers
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )
        and layer.trainable
    ]

    assert not batchnorm_trainable, (
        "BatchNormalization layers must remain frozen."
    )

    print(
        f"Trainable backbone layers: "
        f"{len(trainable_layers)}"
    )

    print(
        f"Trainable weight-bearing layers: "
        f"{len(trainable_weight_layers)}"
    )

    print(
        "BatchNormalization layers trainable: 0"
    )

    return (
        len(trainable_layers),
        len(trainable_weight_layers),
    )