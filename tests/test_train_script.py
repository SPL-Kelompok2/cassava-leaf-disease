import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("torchvision")

from cassava_es.rules import CLASSES  # noqa: E402
from cassava_es.vision import VisionModel  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_vision.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_vision", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dataset(tmp_path, n_per_class=8, int_labels=True):
    d = tmp_path / "train"
    d.mkdir()
    rows = []
    rng = np.random.RandomState(0)
    for ci, c in enumerate(CLASSES):
        for j in range(n_per_class):
            name = f"{c}_{j}.jpg"
            base = np.full((40, 40, 3), ci * 50, dtype="uint8")
            noise = rng.randint(0, 20, (40, 40, 3), dtype="uint8")
            Image.fromarray(base + noise).save(d / name)
            rows.append(dict(image_id=name, label=ci if int_labels else c.lower(),
                             source=2019 if j % 2 else 2020))
    pd.DataFrame(rows).to_csv(tmp_path / "merged.csv", index=False)
    return tmp_path / "merged.csv", d


@pytest.mark.parametrize("int_labels", [True, False])
def test_pelatihan_ujung_ke_ujung(tmp_path, int_labels):
    csv_path, img_dir = _dataset(tmp_path, int_labels=int_labels)
    out = tmp_path / "ck" / "vision.pt"
    rc = _load_script().main([
        "--csv", str(csv_path), "--img-dir", str(img_dir), "--out", str(out),
        "--model", "resnet18", "--no-pretrained", "--img-size", "32",
        "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
    ])
    assert rc == 0
    for f in ("vision.pt", "vision_metrics.json", "vision_split.csv", "vision_test_preds.csv"):
        assert (out.parent / f).is_file(), f

    split = pd.read_csv(out.parent / "vision_split.csv")
    assert len(split) == 40 and set(split.split) == {"train", "test"}
    assert (split.split == "test").sum() == 5 * round(8 * 0.2)

    m = json.loads((out.parent / "vision_metrics.json").read_text())
    assert m["final_test_metrics"]["n"] == (split.split == "test").sum()
    assert 0.0 <= m["final_test_metrics"]["accuracy"] <= 1.0
    assert len(m["meta"]["history"]) == 1

    preds = list(csv.DictReader(open(out.parent / "vision_test_preds.csv")))
    assert len(preds) == (split.split == "test").sum()
    assert sum(float(preds[0][f"p_{c}"]) for c in CLASSES) == pytest.approx(1.0, abs=1e-4)

    # checkpoint hasil pelatihan dapat dimuat dengan weights_only dan dipakai inferensi
    vm = VisionModel.load(out, device="cpu")
    p = vm.predict_proba(img_dir / "CMD_0.jpg")
    assert list(p) == list(CLASSES) and sum(p.values()) == pytest.approx(1.0, abs=1e-5)


def test_inferensi_sama_dengan_evaluasi_pelatihan(tmp_path):
    """Probabilitas di vision_test_preds.csv == hasil VisionModel.predict_proba (pra-pemrosesan sama)."""
    csv_path, img_dir = _dataset(tmp_path)
    out = tmp_path / "vision.pt"
    assert _load_script().main([
        "--csv", str(csv_path), "--img-dir", str(img_dir), "--out", str(out), "--model", "resnet18",
        "--no-pretrained", "--img-size", "32", "--epochs", "1", "--batch-size", "8", "--num-workers", "0",
    ]) == 0
    vm = VisionModel.load(out, device="cpu")
    row = next(csv.DictReader(open(tmp_path / "vision_test_preds.csv")))
    p = vm.predict_proba(img_dir / row["image_id"])
    for c in CLASSES:
        assert p[c] == pytest.approx(float(row[f"p_{c}"]), abs=2e-5)


def test_label_salah_dihentikan_dengan_error(tmp_path):
    csv_path, img_dir = _dataset(tmp_path)
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(object)
    df.loc[0, "label"] = "bukan-kelas"
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        _load_script().main(["--csv", str(csv_path), "--img-dir", str(img_dir), "--model", "resnet18",
                             "--no-pretrained", "--epochs", "1", "--num-workers", "0",
                             "--out", str(tmp_path / "v.pt")])


def test_model_tidak_ada_mengembalikan_kode_1(tmp_path):
    csv_path, img_dir = _dataset(tmp_path)
    rc = _load_script().main(["--csv", str(csv_path), "--img-dir", str(img_dir), "--model", "model_ngawur",
                              "--no-pretrained", "--epochs", "1", "--num-workers", "0",
                              "--out", str(tmp_path / "v.pt")])
    assert rc == 1
