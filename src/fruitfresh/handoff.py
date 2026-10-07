"""Package everything the teammates need (final evaluation, CNN analysis, prototype).

The hand-off contains models, training histories, the split files and a HANDOFF.md
that explains the input contract. It deliberately contains NO test-set results.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES
from .train import run_dir_for

def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


TEMPLATE = """# Hand-off: Baseline CNN + MobileNetV2 (from Onkar)

## Models
{models_table}

All models: **input = float32 RGB image, values 0..255, shape (H, W, 3)** - pre-processing
(rescaling / MobileNetV2 normalisation) is INSIDE the model. Output = 6 softmax probabilities in
this class order (index = position):

`{classes}`

```python
import numpy as np, tensorflow as tf
model = tf.keras.models.load_model("models/<file>.keras")
size = model.input_shape[1:3]                       # (224, 224) unless noted above
x = tf.image.resize(np.array(pil_image.convert("RGB")), size).numpy()[None].astype("float32")
probs = model.predict(x)[0]                         # shape (6,)
```
Use `tf.image.resize` (bilinear) like training did, and do NOT divide by 255 yourself.

## Split (frozen - do not regenerate)
`split/train.csv, validation.csv, test.csv, split_manifest.json` (time-block split: all augmented
copies of a photo stay together and different splits are >= {buffer} s apart in capture time).
`load_split()` in `fruitfresh/splits.py` checks the SHA-256 of the CSVs.

* **The TEST split has not been used for anything** - not for training, early stopping, or choosing
  between experiments. Everything below is VALIDATION only.
* For the final test evaluation use ONE original image per source photo as the primary view
  (`fruitfresh.evaluate.original_only`) and compute confidence intervals by resampling source photos.
  Evaluating all test images (originals + augmented copies) is a secondary, optional view.

## Validation results of the delivered runs (from `experiments.csv`)
{val_table}

## Files
* `models/*.keras` - trained models (best validation loss)
* `runs/<name>/history.csv`, `train_info.json` - per-epoch curves, settings, trainable-layer reports
* `experiments.csv` - all MobileNetV2 experiments (validation metrics)
"""


def build_handoff(work_dir, delivered: dict, seed: int, out_dir, split_dir, experiments_csv=None) -> Path:
    """delivered = {"baseline_cnn": "baseline_cnn", "mobilenet_v2_final": "<run_name of best run>"}"""
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    rows, val_rows = [], []
    for alias, run in delivered.items():
        run_dir = run_dir_for(work_dir, run, seed)
        info = json.loads((run_dir / "train_info.json").read_text())
        shutil.copy(run_dir / "model.keras", out / "models" / f"{alias}.keras")
        (out / "runs" / alias).mkdir(parents=True, exist_ok=True)
        for f in ("history.csv", "train_info.json"):
            shutil.copy(run_dir / f, out / "runs" / alias / f)
        vm = json.loads((run_dir / "val_eval" / "metrics.json").read_text()) if (run_dir / "val_eval" / "metrics.json").exists() else None
        size = tuple(info["image_size"])
        rows.append({"file": f"models/{alias}.keras", "source run": run, "input size": f"{size[0]}x{size[1]}",
                     "params (M)": round(info["params"] / 1e6, 2), "overrides": json.dumps(info["overrides"])})
        if vm:
            val_rows.append({"model": alias, "val accuracy": round(vm["accuracy"], 4), "val macro-F1": round(vm["macro_f1"], 4)})
    split_out = out / "split"
    split_out.mkdir(exist_ok=True)
    for f in Path(split_dir).glob("*"):
        shutil.copy(f, split_out / f.name)
    manifest = json.loads((split_out / "split_manifest.json").read_text())
    if experiments_csv and Path(experiments_csv).exists():
        shutil.copy(experiments_csv, out / "experiments.csv")
    text = TEMPLATE.format(
        models_table=_md_table(pd.DataFrame(rows)) if rows else "",
        classes=", ".join(CLASS_NAMES),
        buffer=manifest["params"].get("buffer_seconds", "?"),
        val_table=_md_table(pd.DataFrame(val_rows)) if val_rows else "(run notebook 05 after the validation evaluation)",
    )
    (out / "HANDOFF.md").write_text(text)
    return out
