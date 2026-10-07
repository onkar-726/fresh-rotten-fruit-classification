"""Model-structure tests (no dataset, no downloads)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from fruitfresh.config import MODEL_SPECS, NUM_CLASSES  # noqa: E402
from fruitfresh.datasets import compute_class_weights  # noqa: E402
from fruitfresh.models import build_model, set_trainable_stage  # noqa: E402

SIZE = (64, 64)


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_model_builds_and_freezes(name):
    model = build_model(name, image_size=SIZE, pretrained=False)
    assert model.output_shape == (None, NUM_CLASSES)
    x = np.random.uniform(0, 255, (2, *SIZE, 3)).astype("float32")
    assert np.allclose(model.predict(x, verbose=0).sum(1), 1, atol=1e-4)
    if MODEL_SPECS[name]["kind"] == "transfer":
        head = set_trainable_stage(model, "head")
        assert head["backbone_layers_trainable"] == 0 and head["trainable_params"] > 0
        ft = set_trainable_stage(model, "finetune", ft_layers=20)
        assert 0 < ft["backbone_weighted_layers_trainable"] <= ft["backbone_layers_trainable"] <= 20
        assert ft["backbone_batchnorm_trainable"] == 0          # BatchNorm must stay frozen
        assert ft["trainable_params"] > head["trainable_params"]
    else:
        rep = set_trainable_stage(model, "all")
        assert rep["trainable_params"] > 0


@pytest.mark.parametrize("alpha", [1.0, 0.75])
def test_mobilenet_alpha_changes_size(alpha):
    full = build_model("mobilenet_v2", image_size=SIZE, pretrained=False, alpha=1.0)
    slim = build_model("mobilenet_v2", image_size=SIZE, pretrained=False, alpha=alpha)
    assert slim.count_params() <= full.count_params()


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_one_training_step(name):
    model = build_model(name, image_size=SIZE, pretrained=False)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    x = np.random.uniform(0, 255, (8, *SIZE, 3)).astype("float32")
    y = np.random.randint(0, NUM_CLASSES, 8)
    assert np.isfinite(model.train_on_batch(x, y)[0])


def test_class_weights_balance():
    w = compute_class_weights(np.array([0] * 80 + [1] * 20))
    assert w[1] > w[0] and np.isclose(80 * w[0], 20 * w[1])
