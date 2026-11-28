"""Decision engine: weighted majority voting, Eq. (1) paper.

    Prediction = argmax_c  sum_i  w_i * p_ic ,   i in {vision, semantic}

Contoh numerik pada paper (Bagian 3.2.3) memperlihatkan hasil yang SUDAH
dinormalisasi dengan jumlah bobot (0.3*0.21 + 0.8*0.49)/1.1 = 0.414, jadi
fungsi ini menghitung rata-rata berbobot (sama seperti ``np.average`` pada kode
decision engine repositori asli). Argmax mengambil kelas pertama bila seri,
sama seperti ``np.argmax``. Verifikasi di tests/test_decision.py.

Paper tidak mengatur skor seri. Karena itu ``top_classes`` melaporkan SEMUA kelas yang
berbagi skor tertinggi, supaya antarmuka dapat menyatakan hasil "belum konklusif"
(penolakan karena ambiguitas; Hendrickx dkk., 2024) alih-alih menampilkan kelas pertama
seolah-olah menang. Label argmax tetap dikembalikan agar perilaku Eq. (1) tidak berubah.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from .rules import CLASSES

# Selisih skor yang dianggap sama (menyerap galat pembulatan floating point).
TIE_TOLERANCE = 1e-9


def top_classes(probs: Mapping[str, float], tol: float = TIE_TOLERANCE) -> list:
    """Semua kelas dengan skor tertinggi (urutan CLASSES). Panjang > 1 berarti seri."""
    best = max(probs[c] for c in CLASSES)
    return [c for c in CLASSES if best - probs[c] <= tol]


def validate_vision_vector(vector: Mapping[str, float]) -> dict:
    """Periksa vektor probabilitas visi: lengkap, berhingga, dalam [0, 1], tidak nol semua.

    Jumlahnya sengaja tidak dipaksa 1 karena contoh numerik paper memakai vektor yang
    tidak berjumlah 1. ``KeyError`` bila ada kelas hilang, ``ValueError`` bila nilai salah.
    """
    missing = [c for c in CLASSES if c not in vector]
    if missing:
        raise KeyError(f"vektor visi kekurangan kelas: {missing}")
    out = {c: float(vector[c]) for c in CLASSES}
    for c, v in out.items():
        if not math.isfinite(v) or v < -TIE_TOLERANCE or v > 1 + 1e-6:
            raise ValueError(f"probabilitas visi {c} harus di antara 0 dan 1, didapat {v}")
    if sum(out.values()) <= 0:
        raise ValueError("vektor visi bernilai nol semua; isi minimal satu probabilitas lebih dari 0")
    return out


def combine(
    semantic: Mapping[str, float],
    vision: Optional[Mapping[str, float]] = None,
    w_semantic: float = 0.5,
    w_vision: float = 0.5,
) -> tuple[dict, str]:
    """Gabungkan probabilitas model semantik dan (opsional) model visi.

    Bila ``vision`` None, hasil = vektor semantik apa adanya.
    Mengembalikan (vektor gabungan per kelas, label argmax).
    """
    if not (math.isfinite(w_semantic) and math.isfinite(w_vision)):
        raise ValueError("bobot harus berupa angka berhingga")
    if w_semantic < 0 or w_vision < 0:
        raise ValueError("bobot tidak boleh negatif")
    if vision is None:
        probs = {c: float(semantic[c]) for c in CLASSES}
    else:
        total = w_semantic + w_vision
        if total <= 0:
            raise ValueError("jumlah bobot harus > 0")
        probs = {
            c: (w_semantic * float(semantic[c]) + w_vision * float(vision[c])) / total
            for c in CLASSES
        }
    return probs, top_classes(probs)[0]
