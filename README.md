# Fresh vs Rotten Fruit Classification - Onkar's part

Capstone project, Deep Learning Lab, Ramdeobaba University, Nagpur (Group 19).
Six classes: fresh / rotten × apple / banana / orange.

## Scope of this repository (Onkar's role only)

| Included (Onkar) | NOT included (other roles) |
|---|---|
| Dataset index + leakage-safe split (shared foundation, needed to train) | Final test-set evaluation and model comparison, confidence intervals - **Rucha** |
| **Baseline CNN** - build, sanity-check, train, validation diagnostics | Prototype app, project report/documentation - **Rucha** |
| **MobileNetV2 transfer learning** - two training stages, layer-freezing checks | CNN variants and CNN-focused analysis (incl. Grad-CAM) - **Krishna** |
| **MobileNetV2 experiments** (10 variants, judged on validation) | |
| Hand-off package for Krishna and Rucha | |

Everything is judged on the **validation** split. The **test split is never used** here, so Rucha's
final evaluation stays clean.

## Quick start (Google Colab)

1. Put this folder on GitHub (or upload the unzipped folder to `/content/fruit-freshness-classifier`).
2. In each notebook's first cell set `REPO_URL` (leave `""` if you uploaded the folder).
3. Use a **GPU** runtime for notebooks 02-04. Run in order:

| Notebook | What it does |
|---|---|
| `01_data_and_split` | download data, build the index, create + verify + save the split (commit `data/splits/`) |
| `02_baseline_cnn` | baseline CNN: overfit-a-tiny-batch check, training, curves, validation report |
| `03_mobilenetv2` | MobileNetV2 stage 1 (head) → stage 2 (fine-tune), proof of what changed, validation report |
| `04_mobilenetv2_experiments` | 10 one-change experiments, validation table, best run chosen by a fixed rule |
| `05_handoff` | package models + split + `HANDOFF.md` for Krishna and Rucha |

Everything is saved **locally** in `workspace/` inside the project - no Google Drive is used.
Finished runs are detected and **loaded instead of retrained** (`FORCE = True` retrains).

**Colab note:** the Colab disk is erased when the runtime resets. Each notebook ends with a backup cell
(`backup_workspace`) that downloads a zip to your own computer; after a reset, set `RESTORE = True` in
the cell below the setup cell and upload that zip. On your own PC nothing extra is needed.

## One-command run (any machine with TensorFlow)

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --experiments head_only ft_top20     # or: --experiments all
pytest                                                              # 16 unit tests
```

## Layout

```
src/fruitfresh/  config.py       all settings + the experiment list
                 data_index.py   file-name parsing, image index, duplicate removal
                 splits.py       time-block split, verify_split, save/load with SHA-256 check
                 datasets.py     tf.data pipeline, augmentation, class weights
                 models.py       baseline CNN, MobileNetV2, freeze/unfreeze, verify_finetuning
                 train.py        training with stages, resume support, per-run train_info.json
                 evaluate.py     validation diagnostics only
                 experiments.py  run/compare/pick the best MobileNetV2 variant
                 handoff.py      build the package for teammates
notebooks/       01..05 (above)
scripts/         run_pipeline.py
tests/           unit tests + synthetic dataset generator
docs/            TRAINING_GUIDE.md (read before the viva)
data/splits/     created by notebook 01 - commit the CSVs and split_manifest.json
results/         small result files from notebooks 04 and 05 - commit them
```

## Why the split is special
The dataset has only ~1,500 original photos; most of the ~13,600 files are augmented **copies**, and the
originals are screenshots taken seconds apart. A random split puts near-identical photos in train and
test, which makes accuracy look far better than it is. The time-block split keeps copies together and
separates train / validation / test by capture time (>= 60 s gap). Details: `docs/TRAINING_GUIDE.md`.

## What has and has not been verified

**Verified** (on the included synthetic stand-in dataset, which has the same folder layout and file-name
conventions as the Kaggle data; smoke mode = 1 epoch per stage, 64 px, no ImageNet weights):

* all 16 unit tests pass;
* all five notebooks execute top to bottom without errors (notebook 04 ran its five cheap experiments);
* `scripts/run_pipeline.py` runs end to end with `head_only ft_top20 alpha_0.75 size_160`; the `alpha_0.75`
  run has 1.39 M parameters vs 2.27 M and the `size_160` run records its 160 px input;
* resume logic: re-running skips every finished run; unknown experiment names are rejected;
* the fine-tuning layer report behaves as designed (more trainable parameters for top-20 < top-40 < top-80,
  **0 trainable BatchNorm layers** in MobileNetV2);
* the training code learns: a 6-epoch baseline on the synthetic data reached 100 % validation accuracy
  (chance = 17 %) - this only shows the code works, the synthetic images are trivially easy;
* the hand-off package builds and contains no test-set results.

**Not verified here** (needs your Colab runs): the real Kaggle download, ImageNet-weight download, GPU timing,
any real accuracy number, and the experiments `no_class_weights`, `no_augmentation`, `dropout_0.5`
(they use the same override mechanism as the verified ones but were not executed).
This repo ships **no results**: `data/splits/` and `results/` are empty until you run the notebooks.
