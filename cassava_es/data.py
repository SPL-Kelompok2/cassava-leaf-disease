"""Pemuatan dataset citra singkong dari ``merged.csv`` (tanpa dependensi torch).

Format: satu folder citra DATAR (bukan sub-folder per penyakit) dan satu CSV berkolom
``image_id``, ``label``, ``source`` (tahun).

Label berupa NAMA (``cbb``, ``CBSD``, ``cassava_mosaic_disease``, ``healthy`` ...) atau
ANGKA 0-4 (konvensi Kaggle 2020, sama dengan urutan kelas pada Eq. 1 paper):

    0 = CBB, 1 = CBSD, 2 = CGM, 3 = CMD, 4 = Healthy

Distribusi kelas dicetak saat memuat untuk memeriksa pemetaan (paper, Gambar 7: CBB 1553,
CBSD 3632, CGM 3159, CMD 15816, Healthy 2893 pada 27 053 citra).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .rules import CLASSES

CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Alias nama label -> kode kelas. Kunci dinormalkan (huruf kecil, tanpa
# tanda baca/spasi) oleh ``_key``.
_ALIASES = {
    "cbb": "CBB", "cassavabacterialblight": "CBB", "bacterialblight": "CBB",
    "cbsd": "CBSD", "cassavabrownstreakdisease": "CBSD", "brownstreakdisease": "CBSD",
    "brownstreak": "CBSD",
    "cgm": "CGM", "cassavagreenmite": "CGM", "cassavagreenmottle": "CGM",
    "greenmite": "CGM", "greenmottle": "CGM",
    "cmd": "CMD", "cassavamosaicdisease": "CMD", "mosaicdisease": "CMD", "mosaic": "CMD",
    "healthy": "Healthy", "sehat": "Healthy",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _key(text: str) -> str:
    return re.sub(r"[^a-z]", "", str(text).strip().lower())


def normalize_label(value) -> str:
    """Ubah satu nilai label (nama atau angka 0-4) menjadi kode kelas.

    Mengangkat ``ValueError`` bila tidak dikenali, supaya salah pemetaan
    tidak lolos diam-diam.
    """
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, float) and float(value).is_integer()
    ):
        idx = int(value)
        if 0 <= idx < len(CLASSES):
            return CLASSES[idx]
        raise ValueError(f"label angka di luar 0-{len(CLASSES) - 1}: {value!r}")
    s = str(value).strip()
    if re.fullmatch(r"\d+", s):
        return normalize_label(int(s))
    k = _key(s)
    if k in _ALIASES:
        return _ALIASES[k]
    raise ValueError(f"label tidak dikenali: {value!r}")


def resolve_image_path(img_dir: Path, image_id: str, source=None) -> Optional[Path]:
    """Cari berkas citra untuk ``image_id``.

    Urutan: nama persis; nama + ekstensi umum; lalu hal yang sama di sub-folder
    bernama ``source`` (jika folder tersebut ada). Mengembalikan None bila
    tidak ketemu.
    """
    bases = [img_dir]
    if source is not None and (img_dir / str(source)).is_dir():
        bases.append(img_dir / str(source))
    for base in bases:
        p = base / str(image_id)
        if p.is_file():
            return p
        for ext in IMAGE_EXTENSIONS:
            q = base / f"{image_id}{ext}"
            if q.is_file():
                return q
    return None


@dataclass
class DatasetReport:
    total_rows: int
    usable: int
    missing_files: int
    duplicated_ids: int
    class_counts: dict
    source_counts: dict
    missing_examples: list


def load_merged_csv(
    csv_path: str | Path,
    img_dir: str | Path,
    *,
    id_col: str = "image_id",
    label_col: str = "label",
    source_col: str = "source",
    check_files: bool = True,
    limit: Optional[int] = None,
    seed: int = 2023,
) -> tuple[pd.DataFrame, DatasetReport]:
    """Muat ``merged.csv`` -> DataFrame [image_id, label, y, source, path] + laporan.

    * ``label``: kode kelas (CBB/CBSD/CGM/CMD/Healthy); ``y``: indeks 0-4.
    * Baris yang berkas citranya tidak ditemukan DIBUANG dan dilaporkan.
    * ``limit``: ambil subset acak (untuk uji cepat/debug).
    """
    csv_path, img_dir = Path(csv_path), Path(img_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Folder citra tidak ditemukan: {img_dir}")
    raw = pd.read_csv(csv_path)
    for col in (id_col, label_col):
        if col not in raw.columns:
            raise KeyError(f"kolom '{col}' tidak ada di CSV (kolom tersedia: {list(raw.columns)})")
    df = pd.DataFrame({
        "image_id": raw[id_col].astype(str),
        "label": [normalize_label(v) for v in raw[label_col]],
        "source": raw[source_col] if source_col in raw.columns else None,
    })
    df["y"] = df["label"].map(CLASS_TO_INDEX).astype(int)
    total = len(df)
    dup = int(df["image_id"].duplicated().sum())

    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    paths, missing = [], []
    if check_files:
        for iid, src in zip(df["image_id"], df["source"]):
            p = resolve_image_path(img_dir, iid, src)
            paths.append(None if p is None else str(p))
            if p is None:
                missing.append(iid)
        df["path"] = paths
        df = df[df["path"].notna()].reset_index(drop=True)
    else:
        df["path"] = [str(img_dir / i) for i in df["image_id"]]

    counts = {c: int((df["label"] == c).sum()) for c in CLASSES}
    src_counts = (
        {str(k): int(v) for k, v in df["source"].value_counts().items()}
        if df["source"].notna().any() else {}
    )
    report = DatasetReport(
        total_rows=total, usable=len(df), missing_files=len(missing), duplicated_ids=dup,
        class_counts=counts, source_counts=src_counts, missing_examples=missing[:5],
    )
    return df, report


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 2023
) -> pd.DataFrame:
    """Tambah kolom ``split`` ('train'/'test') dengan pembagian terstratifikasi.

    Pembagian 80 % latih / 20 % uji (Bagian 4.5 paper), distratifikasi per kelas
    agar proporsi kelas pada data uji sama dengan data latih.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size harus di antara 0 dan 1")
    rng = np.random.RandomState(seed)
    split = pd.Series("train", index=df.index)
    for cls, idx in df.groupby("label").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        if len(idx) > 1:
            n_test = min(max(n_test, 1), len(idx) - 1)
        split.loc[idx[:n_test]] = "test"
    out = df.copy()
    out["split"] = split
    return out


# ------------------------------------------------------------------ metrik
def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Akurasi, presisi/recall/F1 per kelas, 'akurasi per kelas' (one-vs-rest,
    seperti kolom Accuracy pada Tabel 7 paper), dan matriks konfusi.

    Dihitung dengan numpy agar tidak butuh scikit-learn.
    """
    k = len(CLASSES)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    n = int(cm.sum())
    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = dict(
            precision=prec, recall=rec, f1=f1,
            accuracy_ovr=(tp + tn) / n if n else 0.0, support=tp + fn,
        )
    return dict(
        accuracy=float(np.trace(cm) / n) if n else 0.0,
        n=n,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        labels=list(CLASSES),
    )
