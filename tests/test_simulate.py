import random

import pytest

from cassava_es.rules import rules_for
from cassava_es.simulate import HEALTHY_RANGE, TABLE1, simulate, table1_probability


def test_probabilitas_tabel1_terverifikasi():
    """Kolom Probability Tabel 1 = (Upper-Lower)/(Max-Min), setelah dipotong ke [Min,Max]."""
    rows = 0
    for disease, sensors in TABLE1.items():
        for prop, (lo, up, vmin, vmax, p) in sensors.items():
            assert table1_probability(lo, up, vmin, vmax) == pytest.approx(p, abs=0.01), (disease, prop)
            rows += 1
    assert rows == 13


def test_sampel_dalam_rentang_dan_reprodusibel():
    a = simulate("CMD", random.Random(1))
    b = simulate("CMD", random.Random(1))
    assert a == b
    for prop, (_l, _u, vmin, vmax, _p) in TABLE1["CMD"].items():
        assert vmin <= a.values[prop] <= vmax
    for prop in ("SoilPH",):
        lo, hi = HEALTHY_RANGE[prop]
        assert lo <= a.values[prop] <= hi


@pytest.mark.parametrize("label", ["CBB", "CBSD", "CGM", "CMD"])
def test_sampel_favourable_memenuhi_semua_aturan(label):
    rng = random.Random(7)
    for _ in range(25):
        s = simulate(label, rng, favourable=True)
        for r in rules_for(label):
            v = s.rainy_season if r.observed_property is None else s.values[r.observed_property]
            assert r.satisfied_by(v), (label, r.rule_id, v)
