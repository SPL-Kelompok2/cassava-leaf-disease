import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cassava_es.data import (
    classification_metrics,
    load_merged_csv,
    normalize_label,
    resolve_image_path,
    stratified_split,
)
from cassava_es.rules import CLASSES


# ------------------------------------------------------------------ label
@pytest.mark.parametrize("raw,expected", [
    ("cbb", "CBB"), ("CBB", "CBB"), ("Cassava Bacterial Blight", "CBB"),
    ("cbsd", "CBSD"), ("cassava_brown_streak_disease", "CBSD"),
    ("cgm", "CGM"), ("Cassava Green Mite", "CGM"), ("cassava green mottle", "CGM"),
    ("cmd", "CMD"), ("Cassava-Mosaic-Disease", "CMD"),
    ("healthy", "Healthy"), ("Healthy", "Healthy"),
    (0, "CBB"), (1, "CBSD"), (2, "CGM"), (3, "CMD"), (4, "Healthy"),
    ("3", "CMD"), (np.int64(4), "Healthy"), (2.0, "CGM"),
])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("bad", ["unknown", "", 5, -1, "7", 1.5, None])
def test_normalize_label_tidak_dikenal(bad):
    with pytest.raises(ValueError):
        normalize_label(bad)


def test_urutan_kelas_angka_sama_dengan_konvensi_kaggle_2020():
    # 0=CBB, 1=CBSD, 2=CGM, 3=CMD, 4=Healthy
    assert [normalize_label(i) for i in range(5)] == ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    assert list(CLASSES) == ["CBB", "CBSD", "CGM", "CMD", "Healthy"]


# ------------------------------------------------------------------ muat csv
def _make_dataset(tmp_path, labels, *, ext=".jpg", id_has_ext=True, source="2019", skip=()):
    img_dir = tmp_path / "train"
    img_dir.mkdir()
    rows = []
    for i, lab in enumerate(labels):
        name = f"img_{i}"
        fname = name + ext
        if i not in skip:
            Image.fromarray(np.full((8, 8, 3), i * 5 % 255, dtype="uint8")).save(img_dir / fname)
        rows.append(dict(image_id=fname if id_has_ext else name, label=lab, source=source))
    csv = tmp_path / "merged.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, img_dir


def test_muat_csv_label_nama(tmp_path):
    labs = ["cbb", "cbsd", "cgm", "cmd", "healthy"] * 4
    csv, d = _make_dataset(tmp_path, labs)
    df, rep = load_merged_csv(csv, d)
    assert len(df) == 20 and rep.missing_files == 0 and rep.usable == 20
    assert rep.class_counts == {c: 4 for c in CLASSES}
    assert set(df.columns) >= {"image_id", "label", "y", "source", "path"}
    assert df.loc[df.label == "CMD", "y"].unique().tolist() == [3]


def test_muat_csv_label_angka(tmp_path):
    csv, d = _make_dataset(tmp_path, [0, 1, 2, 3, 4, 3, 3])
    df, rep = load_merged_csv(csv, d)
    assert rep.class_counts["CMD"] == 3 and rep.class_counts["CBB"] == 1


def test_id_tanpa_ekstensi_ditemukan(tmp_path):
    csv, d = _make_dataset(tmp_path, ["cmd"] * 3, id_has_ext=False)
    df, rep = load_merged_csv(csv, d)
    assert rep.usable == 3 and all(p.endswith(".jpg") for p in df.path)


def test_berkas_hilang_dibuang_dan_dilaporkan(tmp_path):
    csv, d = _make_dataset(tmp_path, ["cmd"] * 5, skip=(1, 3))
    df, rep = load_merged_csv(csv, d)
    assert rep.total_rows == 5 and rep.usable == 3 and rep.missing_files == 2
    assert len(rep.missing_examples) == 2


def test_subfolder_source(tmp_path):
    d = tmp_path / "train"
    (d / "2019").mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype="uint8")).save(d / "2019" / "a.jpg")
    assert resolve_image_path(d, "a.jpg", "2019") is not None
    assert resolve_image_path(d, "a.jpg", "2020") is None


def test_id_ganda_dilaporkan(tmp_path):
    csv, d = _make_dataset(tmp_path, ["cmd", "cmd"])
    df = pd.read_csv(csv)
    df.loc[1, "image_id"] = df.loc[0, "image_id"]
    df.to_csv(csv, index=False)
    _, rep = load_merged_csv(csv, d)
    assert rep.duplicated_ids == 1


def test_label_salah_memicu_error(tmp_path):
    csv, d = _make_dataset(tmp_path, ["cmd", "bukan-penyakit"])
    with pytest.raises(ValueError):
        load_merged_csv(csv, d)


def test_kolom_hilang_dan_path_salah(tmp_path):
    csv, d = _make_dataset(tmp_path, ["cmd"] * 2)
    with pytest.raises(KeyError):
        load_merged_csv(csv, d, label_col="tidak_ada")
    with pytest.raises(FileNotFoundError):
        load_merged_csv(tmp_path / "x.csv", d)
    with pytest.raises(FileNotFoundError):
        load_merged_csv(csv, tmp_path / "tidak-ada")


def test_limit_mengambil_subset(tmp_path):
    csv, d = _make_dataset(tmp_path, ["cmd"] * 10)
    df, _ = load_merged_csv(csv, d, limit=4)
    assert len(df) == 4


# ------------------------------------------------------------------ split
def test_split_terstratifikasi_deterministik():
    labs = (["cbb"] * 30) + (["cbsd"] * 50) + (["cgm"] * 40) + (["cmd"] * 200) + (["healthy"] * 20)
    df = pd.DataFrame({"image_id": range(len(labs)), "label": [normalize_label(x) for x in labs]})
    a, b = stratified_split(df, 0.2, seed=1), stratified_split(df, 0.2, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != stratified_split(df, 0.2, seed=2)["split"].tolist()
    for c, n in zip(CLASSES, [30, 50, 40, 200, 20]):
        sub = a[a.label == c]
        assert (sub.split == "test").sum() == round(n * 0.2)
        assert (sub.split == "train").sum() == n - round(n * 0.2)


def test_split_kelas_kecil_tetap_punya_latih_dan_uji():
    df = pd.DataFrame({"image_id": range(3), "label": ["CBB"] * 3})
    out = stratified_split(df, 0.2)
    assert set(out.split) == {"train", "test"}


def test_split_test_size_tidak_valid():
    with pytest.raises(ValueError):
        stratified_split(pd.DataFrame({"label": ["CBB"]}), 1.5)


# ------------------------------------------------------------------ metrik
# Matriks konfusi VOLO-D2@384 dari Gambar 10 paper (baris = label benar,
# kolom = prediksi; urutan CBB, CBSD, CGM, CMD, Healthy).
PAPER_CM = np.array([
    [195, 26, 10, 10, 58],
    [32, 593, 12, 42, 16],
    [6, 18, 484, 62, 34],
    [11, 14, 29, 3014, 24],
    [35, 35, 25, 47, 436],
])


def _from_cm(cm):
    yt, yp = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            yt += [i] * int(cm[i, j])
            yp += [j] * int(cm[i, j])
    return yt, yp


def test_metrik_cocok_dengan_gambar10_dan_tabel7_paper():
    m = classification_metrics(*_from_cm(PAPER_CM))
    assert m["n"] == 5268  # = jumlah sampel uji pada Gambar 17 paper
    assert m["accuracy"] == pytest.approx(0.8964, abs=5e-5)  # Tabel 6
    table7 = {  # presisi, recall, akurasi (one-vs-rest), F1
        "CBB": (0.699, 0.652, 0.964, 0.675),
        "CBSD": (0.864, 0.853, 0.963, 0.859),
        "CGM": (0.864, 0.801, 0.963, 0.831),
        "CMD": (0.949, 0.975, 0.955, 0.962),
        "Healthy": (0.768, 0.754, 0.948, 0.761),
    }
    for c, (p, r, a, f) in table7.items():
        got = m["per_class"][c]
        assert got["precision"] == pytest.approx(p, abs=6e-4), c
        assert got["recall"] == pytest.approx(r, abs=6e-4), c
        assert got["accuracy_ovr"] == pytest.approx(a, abs=6e-4), c
        # F1 pada Tabel 7 tampaknya dihitung dari presisi & recall yang sudah dibulatkan 3 desimal
        # (mis. CGM: 2*0.864*0.801/1.665 = 0.8313 vs 0.8316 dari matriks mentah), jadi toleransinya 1e-3.
        assert got["f1"] == pytest.approx(f, abs=1e-3), c


def test_metrik_kasus_kosong_dan_sempurna():
    m = classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert m["accuracy"] == 1.0 and m["per_class"]["CMD"]["f1"] == 1.0
    z = classification_metrics([], [])
    assert z["accuracy"] == 0.0 and z["n"] == 0
