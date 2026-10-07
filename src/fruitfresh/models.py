"""Model definitions (Baseline CNN and MobileNetV2).

Both share the same outer structure, which keeps training, freezing and the
hand-off to teammates identical:

    image (float 0..255) -> [rescaling] -> backbone -> GAP -> Dropout -> Dense(softmax)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .config import IMAGE_SIZE, MODEL_SPECS, NUM_CLASSES


def _baseline_backbone(input_shape) -> keras.Model:
    """4 conv blocks (32-64-128-256), each Conv-BN-ReLU x2 + MaxPool."""
    inp = keras.Input(shape=input_shape)
    x = inp
    for filters in (32, 64, 128, 256):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization(momentum=0.9)(x)  # 0.9 (not 0.99): moving averages catch up within a few epochs
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    return keras.Model(inp, x, name="backbone")


def _mobilenet_backbone(input_shape, pretrained: bool, alpha: float) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, alpha=alpha,
        weights="imagenet" if pretrained else None,
    )
    return keras.Model(base.input, base.output, name="backbone")


def build_model(
    name: str,
    image_size=IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float | None = None,
    alpha: float | None = None,
) -> keras.Model:
    """Build `baseline_cnn` or `mobilenet_v2` (input: float RGB 0..255)."""
    spec = MODEL_SPECS[name]
    dropout = spec["dropout"] if dropout is None else dropout
    alpha = spec.get("alpha", 1.0) if alpha is None else alpha
    input_shape = (*image_size, 3)
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    if spec["preprocess"] == "rescale_255":
        x = layers.Rescaling(1.0 / 255, name="rescale")(x)
    elif spec["preprocess"] == "mobilenet_v2":  # MobileNetV2 expects [-1, 1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)

    if spec["kind"] == "scratch":
        backbone = _baseline_backbone(tuple(x.shape[1:]))
        x = backbone(x)
    else:
        backbone = _mobilenet_backbone(input_shape, pretrained, alpha)
        x = backbone(x, training=False)  # BatchNorm stays in inference mode, even when fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)


def set_trainable_stage(model: keras.Model, stage: str, ft_layers: int = 40) -> dict:
    """Freeze / unfreeze layers for a training stage and return a report.

    stage = "all"       : everything trainable (baseline CNN)
            "head"      : backbone frozen, only GAP/Dropout/Dense train
            "finetune"  : last ``ft_layers`` backbone layers train, but
                          BatchNorm layers always stay frozen
    """
    backbone = model.get_layer("backbone")
    if stage == "all":
        model.trainable = True
    elif stage == "head":
        backbone.trainable = False
    elif stage == "finetune":
        backbone.trainable = True
        for layer in backbone.layers[:-ft_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return trainable_report(model)


def trainable_report(model: keras.Model) -> dict:
    bb = model.get_layer("backbone")
    return {
        "trainable_params": int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
        "total_params": int(model.count_params()),
        "backbone_layers": len(bb.layers),
        "backbone_layers_trainable": int(sum(l.trainable for l in bb.layers)),
        "backbone_weighted_layers_trainable": int(sum(l.trainable and len(l.weights) > 0 for l in bb.layers)),
        "backbone_batchnorm_trainable": int(
            sum(isinstance(l, layers.BatchNormalization) and l.trainable for l in bb.layers)
        ),
    }


def verify_finetuning(stage1_path, final_path) -> dict:
    """Compare backbone weights before/after stage 2.

    Proves WHICH layers fine-tuning really changed (and that BatchNorm did not change).
    """
    b1 = keras.models.load_model(Path(stage1_path)).get_layer("backbone")
    b2 = keras.models.load_model(Path(final_path)).get_layer("backbone")
    changed, bn_changed = [], 0
    for i, (l1, l2) in enumerate(zip(b1.layers, b2.layers)):
        w1, w2 = l1.get_weights(), l2.get_weights()
        if w1 and any(not np.allclose(a, b, atol=1e-8) for a, b in zip(w1, w2)):
            changed.append(i)
            bn_changed += isinstance(l1, layers.BatchNormalization)
    return {
        "backbone_layers": len(b1.layers),
        "layers_with_changed_weights": len(changed),
        "first_changed_layer": min(changed) if changed else None,
        "last_changed_layer": max(changed) if changed else None,
        "batchnorm_layers_changed": int(bn_changed),
    }
