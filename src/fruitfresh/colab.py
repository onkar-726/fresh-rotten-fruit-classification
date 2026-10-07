"""Helpers so the same notebooks run on a laptop and in Google Colab.

EVERYTHING IS SAVED LOCALLY - no Google Drive is used or needed.

* Laptop / own PC : files go to ``<project>/workspace`` and stay there.
* Colab           : the same folder lives on the temporary runtime disk, which is ERASED when the
                    runtime resets. Use ``backup_workspace()`` to download a zip to your own computer
                    and ``restore_workspace()`` to upload it again next time.
"""
from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SKIP_IN_BACKUP = ("checkpoint_",)  # per-stage checkpoints are duplicates of the final models


def in_colab() -> bool:
    return "google.colab" in sys.modules


def setup_workspace() -> Path:
    """Local folder for models, histories and predictions.

    FRUIT_WORK_DIR (environment variable) wins; otherwise ``<project>/workspace``.
    """
    path = Path(os.getenv("FRUIT_WORK_DIR") or PROJECT_ROOT / "workspace")
    path.mkdir(parents=True, exist_ok=True)
    return path


def backup_workspace(work_dir, name: str = "fruit_workspace_backup", download: bool = True, exclude=()) -> Path:
    """Zip the workspace (models, histories, split) next to it; in Colab also download it to your PC.

    ``exclude`` = path fragments to leave out, e.g. ``exclude=("runs/ft_top80", "runs/size_160")``
    to keep the zip small (every run's model is roughly 25-30 MB).
    """
    work_dir = Path(work_dir)
    out = work_dir.parent / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work_dir.rglob("*")):
            rel = f.relative_to(work_dir).as_posix()
            if f.is_file() and not f.name.startswith(_SKIP_IN_BACKUP) and not any(x in rel for x in exclude):
                z.write(f, rel)
    print(f"Backup written: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if download and in_colab():
        from google.colab import files  # saves the zip through the browser to your own computer

        files.download(str(out))
    return out


def restore_workspace(work_dir, zip_path=None) -> Path:
    """Unpack a backup zip into the workspace. In Colab, ``zip_path=None`` opens an upload dialog."""
    work_dir = Path(work_dir)
    if zip_path is None:
        if not in_colab():
            raise ValueError("Pass zip_path=... (the backup zip) when not running in Colab.")
        from google.colab import files

        zip_path = next(iter(files.upload()))
    shutil.unpack_archive(str(zip_path), str(work_dir))
    print(f"Restored {zip_path} -> {work_dir}")
    return work_dir


def gpu_summary() -> str:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return f"TensorFlow {tf.__version__} | GPUs: {[g.name for g in gpus] or 'none (CPU only)'}"
