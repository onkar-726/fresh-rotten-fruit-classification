"""Run Onkar's whole part headless:
index -> split -> baseline CNN -> MobileNetV2 (2 stages) -> experiments -> hand-off zip.

    python scripts/run_pipeline.py --experiments head_only ft_top20
    python scripts/run_pipeline.py --experiments all
    FRUIT_SMOKE=1 FRUIT_NO_PRETRAINED=1 FRUIT_DATASET_ROOT=<fake> python scripts/run_pipeline.py --experiments head_only

The notebooks do the same steps interactively. Everything is judged on VALIDATION;
the test split is never loaded for training or selection.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitfresh import config  # noqa: E402
from fruitfresh.colab import setup_workspace  # noqa: E402
from fruitfresh.data_index import build_index, get_dataset_root, summarize_index  # noqa: E402
from fruitfresh.evaluate import evaluate_validation  # noqa: E402
from fruitfresh.experiments import pick_best, run_mobilenet_experiments, summarize_runs  # noqa: E402
from fruitfresh.handoff import build_handoff  # noqa: E402
from fruitfresh.splits import load_split, make_time_block_split, save_split, verify_split  # noqa: E402
from fruitfresh.train import run_dir_for, train_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=[],
                    help="names from config.MOBILENET_EXPERIMENTS, or 'all' (default: none)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run exists")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline tests only)")
    a = ap.parse_args()

    names = list(config.MOBILENET_EXPERIMENTS) if a.experiments == ["all"] else a.experiments
    unknown = [n for n in names if n not in config.MOBILENET_EXPERIMENTS]
    if unknown:
        sys.exit(f"Unknown experiment(s): {unknown}. Choose from {list(config.MOBILENET_EXPERIMENTS)}")
    pretrained = not (a.no_pretrained or os.getenv("FRUIT_NO_PRETRAINED") == "1")

    work = setup_workspace()
    root = get_dataset_root()
    print("dataset:", root, "| work dir:", work)

    split_dir = work / "data" / "splits"
    if not (split_dir / "split_manifest.json").exists():
        index = build_index(root)
        print(summarize_index(index))
        idx_split, rep = make_time_block_split(index, seed=a.seed, **config.SPLIT_PARAMS)
        stats = verify_split(idx_split, config.SPLIT_PARAMS["buffer_seconds"])
        save_split(idx_split, split_dir, {**config.SPLIT_PARAMS, "seed": a.seed, **rep}, stats)
        print("created split:", stats["images"], stats["sources"])
    else:
        print("re-using existing split:", split_dir)
    train_df, val_df, _ = load_split(split_dir)  # the TEST split is deliberately ignored

    train_model("baseline_cnn", train_df, val_df, root, work, seed=a.seed, force=a.force, pretrained=False)
    run_mobilenet_experiments(names, train_df, val_df, root, work, a.seed, pretrained=pretrained, force=a.force)

    runs = ["mobilenet_v2"] + names
    table = summarize_runs(work, runs, a.seed, val_df, root)
    print(table.sort_values("val_macro_f1", ascending=False).to_string(index=False))
    table.to_csv(work / "experiments.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "mobilenetv2_experiments_validation.csv", index=False)

    best = pick_best(table)["run"]
    print("best MobileNetV2 run on validation:", best)
    evaluate_validation(run_dir_for(work, "baseline_cnn", a.seed), val_df, root)
    out = build_handoff(work, {"baseline_cnn": "baseline_cnn", "mobilenet_v2_final": best}, a.seed,
                        work / "handoff", split_dir, work / "experiments.csv")
    print("hand-off zip:", shutil.make_archive(str(work / "handoff_onkar"), "zip", out))


if __name__ == "__main__":
    main()
