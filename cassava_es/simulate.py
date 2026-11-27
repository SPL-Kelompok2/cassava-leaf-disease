"""Simulator data sensor berdasarkan Tabel 1 paper.

Paper (Bagian 4.1): data sensor disimulasikan dengan distribusi seragam
[Min, Max] per sensor untuk tiap kelas; kolom Probability pada Tabel 1 adalah
peluang sebuah sampel jatuh di dalam rentang aturan [Lower, Upper].
Properti ini diverifikasi di tests/test_simulate.py.

Asumsi implementasi (tidak dirinci pada paper):
 - sensor yang tidak tercantum untuk suatu penyakit diambil dari rentang
   "Healthy" (Tabel 1) secara seragam;
 - musim hujan (boolean) diundi Bernoulli(0.5) untuk semua kelas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# (lower, upper, min, max, probability_pada_paper_dalam_persen)
# lower/upper None = tidak ada batas pada aturan; angka batas atas Tabel 1
# yang "melewati" domain fisik (mis. 1 untuk kelembapan) tetap dicatat apa adanya.
TABLE1 = {
    "CBB": {
        "SoilMoisture": (0.3, 1, 0.1, 1, 77.78),
        "SoilPH": (6.5, 7.2, 6.3, 7.5, 58.33),
        "SoilTemperature": (25, 30, 23, 31, 62.50),
    },
    "CBSD": {
        "RelativeHumidity": (0.7, 0.85, 0.65, 0.88, 65.22),
        "SoilMoisture": (0.1, 1, 0, 1, 90.00),
        "SoilTemperature": (10, 32, 5, 33, 78.57),
    },
    "CGM": {
        "SoilMoisture": (0.7, 1, 0.55, 1, 66.67),
        "RelativeHumidity": (0.7, 1, 0.6, 1, 75.00),
        "SoilTemperature": (27, 40, 24, 40, 81.25),
    },
    "CMD": {
        "Temperature": (30, 50, 24, 50, 76.92),
        "SoilMoisture": (0.3, 1, 0.1, 1, 77.78),
        "SoilTemperature": (20, 32, 18, 34, 75.00),
        "RelativeHumidity": (0.8, 1, 0.7, 1, 66.67),
    },
}

# Healthy: hanya Min-Max (tidak ada Lower/Upper/Probability). Baris "Soil moisture"
# tercetak dua kali di Tabel 1 paper dengan nilai sama (0.2-0.8); dipakai sekali.
HEALTHY_RANGE = {
    "SoilMoisture": (0.2, 0.8),
    "SoilTemperature": (5, 40),
    "RelativeHumidity": (0.2, 0.8),
    "Temperature": (10, 50),
    "SoilPH": (3, 10),
}

SENSORS = list(HEALTHY_RANGE)


@dataclass
class Sample:
    label: str  # kelas skenario (CBB/CBSD/CGM/CMD/Healthy)
    rainy_season: bool
    values: dict  # nama ObservableProperty -> float


def table1_probability(lower, upper, vmin, vmax) -> float:
    """Peluang uniform[vmin, vmax] jatuh di [lower, upper] (dalam persen)."""
    hi = vmax if upper is None else min(upper, vmax)
    lo = vmin if lower is None else max(lower, vmin)
    return 100.0 * max(0.0, hi - lo) / (vmax - vmin)


def simulate(label: str, rng: random.Random, favourable: bool = False,
             max_tries: int = 100000) -> Sample:
    """Satu sampel sensor untuk skenario ``label``.

    ``favourable=True`` (pilihan implementasi ini, bukan paper) mengulang undian
    sampai SEMUA kondisi aturan penyakit ``label`` terpenuhi, agar skenario demo
    informatif. Musim hujan dipaksa True untuk CBB karena R01 membutuhkannya.
    """
    if favourable and label != "Healthy":
        from .rules import rules_for  # impor lokal: hindari impor melingkar
        rules = rules_for(label)
        for _ in range(max_tries):
            sample = simulate(label, rng)
            if label == "CBB":
                sample.rainy_season = True
            ok = all(
                r.satisfied_by(sample.rainy_season if r.observed_property is None
                               else sample.values[r.observed_property])
                for r in rules
            )
            if ok:
                return sample
        raise RuntimeError("tidak menemukan sampel favourable")
    values = {}
    for prop, (lo, hi) in HEALTHY_RANGE.items():
        values[prop] = rng.uniform(lo, hi)
    if label != "Healthy":
        for prop, (_l, _u, vmin, vmax, _p) in TABLE1[label].items():
            values[prop] = rng.uniform(vmin, vmax)
    values = {k: round(v, 3) for k, v in values.items()}
    return Sample(label=label, rainy_season=rng.random() < 0.5, values=values)
