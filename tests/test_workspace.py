import zipfile

from fruitfresh.colab import backup_workspace, restore_workspace, setup_workspace


def test_workspace_is_local_and_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FRUIT_WORK_DIR", str(tmp_path / "w"))
    w = setup_workspace()
    assert w == tmp_path / "w" and w.is_dir()
    assert "drive" not in str(w).lower()


def test_default_workspace_is_inside_project(monkeypatch, tmp_path):
    from fruitfresh import colab

    monkeypatch.delenv("FRUIT_WORK_DIR", raising=False)
    monkeypatch.setattr(colab, "PROJECT_ROOT", tmp_path)  # keeps the real project folder clean
    assert setup_workspace() == tmp_path / "workspace"


def test_backup_restore_roundtrip(tmp_path):
    w = tmp_path / "ws"
    (w / "runs" / "a" / "seed42").mkdir(parents=True)
    (w / "runs" / "a" / "seed42" / "model.keras").write_bytes(b"model")
    (w / "runs" / "a" / "seed42" / "checkpoint_all.keras").write_bytes(b"dup")
    (w / "data").mkdir()
    (w / "data" / "train.csv").write_text("x")
    z = backup_workspace(w, download=False)
    assert "runs/a/seed42/checkpoint_all.keras" not in zipfile.ZipFile(z).namelist()
    new = tmp_path / "fresh"
    restore_workspace(new, z)
    assert (new / "runs" / "a" / "seed42" / "model.keras").read_bytes() == b"model"
    assert (new / "data" / "train.csv").read_text() == "x"


def test_backup_exclude(tmp_path):
    w = tmp_path / "ws"
    for run in ("keep", "skip"):
        (w / "runs" / run).mkdir(parents=True)
        (w / "runs" / run / "model.keras").write_bytes(b"m")
    names = zipfile.ZipFile(backup_workspace(w, download=False, exclude=("runs/skip",))).namelist()
    assert names == ["runs/keep/model.keras"]
