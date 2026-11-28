"""Pemeriksaan citra ganda sebelum membagi data latih/uji (tanpa dependensi torch).

Citra yang sama di data latih dan uji adalah kebocoran data jenis "duplicates in datasets"
(Kapoor & Narayanan, 2023, kategori L1.4): akurasi uji tampak lebih tinggi daripada
kemampuan model pada foto yang benar-benar baru.

Dua tingkat pemeriksaan:

* **identik**  : isi berkas sama persis (SHA-256). Pasti ganda.
* **mirip**    : *difference hash* (dHash) 64 bit berjarak Hamming <= ambang, misalnya foto
  yang sama setelah diubah ukurannya atau dikompresi ulang. Ini KANDIDAT yang perlu dilihat
  mata: foto berbeda dari adegan seragam dapat memiliki hash yang berdekatan, dan hasil
  pemotongan/rotasi tidak terdeteksi.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

HASH_SIZE = 8  # 8 x 8 = 64 bit
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: str | Path, size: int = HASH_SIZE) -> int:
    """Difference hash: bit = apakah piksel kanan lebih terang daripada piksel kirinya."""
    with Image.open(path) as im:
        im.draft("L", (size * 8, size * 8))  # percepat dekode JPEG besar
        small = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def near_pairs(hashes: Sequence[int], max_distance: int, chunk: int = 512) -> list[tuple[int, int, int]]:
    """Semua pasangan (i, j, jarak), i < j, dengan jarak Hamming <= ``max_distance``."""
    h = np.array(list(hashes), dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(h), chunk):
        x = h[start:start + chunk, None] ^ h[None, :]
        dist = _POPCOUNT[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=2, dtype=np.uint16)
        for i, j in zip(*np.nonzero(dist <= max_distance)):
            a = start + int(i)
            if a < j:
                pairs.append((a, int(j), int(dist[i, j])))
    return pairs


def _groups(n: int, pairs) -> np.ndarray:
    """Komponen terhubung (union-find) -> nomor grup per indeks."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def find_duplicates(df: pd.DataFrame, max_distance: int = 4, path_col: str = "path") -> pd.DataFrame:
    """Tambahkan kolom ``sha256``, ``dhash``, ``exact_group``, ``near_group``.

    Nomor grup = indeks baris pertama dalam grup. ``near_group`` sudah mencakup berkas identik.
    """
    if not 0 <= max_distance <= 64:
        raise ValueError("max_distance harus 0-64")
    out = df.reset_index(drop=True).copy()
    out["sha256"] = [sha256_file(p) for p in out[path_col]]
    hashes = [dhash(p) for p in out[path_col]]
    out["dhash"] = [f"{v:016x}" for v in hashes]
    first = {s: i for i, s in reversed(list(enumerate(out["sha256"])))}
    exact = [(first[s], i) for i, s in enumerate(out["sha256"]) if first[s] != i]
    out["exact_group"] = _groups(len(out), exact)
    near = [(a, b) for a, b, _ in near_pairs(hashes, max_distance)]
    out["near_group"] = _groups(len(out), exact + near)
    return out


def summarize(report: pd.DataFrame, group_col: str) -> dict:
    """Ringkasan grup berukuran >= 2: jumlah, berlabel bentrok, dan yang menyeberang split."""
    multi = report[report.groupby(group_col)[group_col].transform("size") > 1]
    g = multi.groupby(group_col)
    res = {
        "groups": int(g.ngroups),
        "images": int(len(multi)),
        "extra_copies": int(len(multi) - g.ngroups),
        "label_conflicts": int((g["label"].nunique() > 1).sum()) if len(multi) and "label" in multi else 0,
    }
    if "split" in report.columns:
        res["cross_split_groups"] = int((g["split"].nunique() > 1).sum()) if len(multi) else 0
    return res


def deduplicate(report: pd.DataFrame, group_col: str = "exact_group") -> pd.DataFrame:
    """Sisakan satu citra per grup; grup yang labelnya bentrok dibuang seluruhnya."""
    conflict = report.groupby(group_col)["label"].transform("nunique") > 1
    return report[~conflict].drop_duplicates(subset=group_col, keep="first")
