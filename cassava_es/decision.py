"""Decision engine: weighted majority voting, Eq. (1) paper.

    Prediction = argmax_c  sum_i  w_i * p_ic ,   i in {vision, semantic}

Contoh numerik pada paper (Bagian 3.2.3) memperlihatkan hasil yang SUDAH
dinormalisasi dengan jumlah bobot (0.3*0.21 + 0.8*0.49)/1.1 = 0.414, jadi
fungsi ini menghitung rata-rata berbobot (sama seperti ``np.average`` pada kode
decision engine repositori asli). Argmax mengambil kelas pertama bila seri,
sama seperti ``np.argmax``. Verifikasi di tests/test_decision.py.
"""

from __future__ import annotations

from typing import Mapping, Optional

from .rules import CLASSES


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
    best = max(CLASSES, key=lambda c: (probs[c], -CLASSES.index(c)))
    return probs, best
