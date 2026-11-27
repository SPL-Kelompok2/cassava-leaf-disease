import itertools
import random

import pytest

from cassava_es.kb import CassavaKB, Scenario, diagnose_one
from cassava_es.rules import DISEASES, RULES, rules_for
from cassava_es.simulate import HEALTHY_RANGE
from conftest import needs_java

pytestmark = needs_java


def _rand_values(rng):
    return {p: round(rng.uniform(lo, hi), 3) for p, (lo, hi) in HEALTHY_RANGE.items()}


def test_pellet_sama_dengan_python_pada_60_skenario_acak():
    """Satu dunia, 60 tanaman: hasCondition hasil Pellet == evaluasi Python semua aturan."""
    rng = random.Random(123)
    kb = CassavaKB()
    for pid in range(1, 61):
        kb.add_scenario(Scenario(pid, rng.random() < 0.5, _rand_values(rng)))
    kb.reason()
    for pid in range(1, 61):
        d = kb.diagnose(pid)
        assert d.consistent, f"plant_{pid} tidak konsisten"
        assert len(d.fired_conditions()) == len(set(d.fired_conditions()))


# nilai tepat di batas + sedikit di dalam/luar (inklusif di kedua ujung)
_BOUNDARY = {
    "SoilMoisture": [0.0999, 0.1, 0.3, 0.7, 0.7001],
    "SoilPH": [6.4999, 6.5, 7.2, 7.2001],
    "SoilTemperature": [9.999, 10, 20, 25, 27, 30, 32, 32.001],
    "RelativeHumidity": [0.6999, 0.7, 0.8, 0.85, 0.8501],
    "Temperature": [29.999, 30, 30.001],
}


def test_nilai_batas_cocok_dengan_python():
    kb = CassavaKB()
    pid = 0
    for prop, values in _BOUNDARY.items():
        for v in values:
            pid += 1
            kb.add_scenario(Scenario(pid, False, {prop: v}))
    kb.reason()
    for p in range(1, pid + 1):
        assert kb.diagnose(p).consistent, p
    assert pid == sum(len(v) for v in _BOUNDARY.values())


def test_rumus_probabilitas_dan_sehat():
    d = diagnose_one(dict(SoilMoisture=0.5, SoilPH=6.8, SoilTemperature=27.0,
                          RelativeHumidity=0.6, Temperature=25.0), rainy_season=True)
    # CBB: semua 4 terpenuhi; CBSD: SM & ST (2/3); CGM: ST (1/3); CMD: SM & ST (2/4)
    assert d.diseases["CBB"].probability == 1.0
    assert d.diseases["CBSD"].probability == pytest.approx(2 / 3)
    assert d.diseases["CGM"].probability == pytest.approx(1 / 3)
    assert d.diseases["CMD"].probability == pytest.approx(0.5)
    assert d.healthy_probability == pytest.approx(1 - 9 / 14)
    assert d.consistent


def test_tanpa_observasi_hanya_musim_hujan():
    d = diagnose_one({}, rainy_season=True)
    assert d.fired_conditions() == ["CBBRainySeason"]
    assert d.healthy_probability == pytest.approx(1 - 1 / 14)
    d2 = diagnose_one({}, rainy_season=False)
    assert d2.fired_conditions() == [] and d2.healthy_probability == 1.0


def test_penjelasan_memakai_definisi_tbox():
    d = diagnose_one(dict(SoilTemperature=27.0), rainy_season=False)
    assert "Soil temperature of 25" in d.diseases["CBB"].explanations[0]
    assert d.diseases["CBB"].symptoms.startswith("The disease usually causes the following visual symptoms")


def test_inferensi_tingkat2_dan_klasifikasi_dl():
    full = diagnose_one(dict(SoilMoisture=0.5, SoilPH=6.8, SoilTemperature=27.0,
                             RelativeHumidity=0.6, Temperature=25.0), rainy_season=True)
    assert full.diseases["CBB"].fully_favourable and "HighRiskPlant" in full.types
    assert ("plant_1", "isFullyFavourableFor", "CassavaBacterialBlight") in full.derived
    none = diagnose_one(dict(SoilMoisture=0.05), rainy_season=False)
    assert not any(r.fully_favourable for r in none.diseases.values())
    assert "HighRiskPlant" not in none.types


def test_fully_favourable_iff_semua_kondisi():
    rng = random.Random(5)
    kb = CassavaKB()
    for pid in range(1, 31):
        kb.add_scenario(Scenario(pid, rng.random() < 0.5, _rand_values(rng)))
    kb.reason()
    for pid in range(1, 31):
        d = kb.diagnose(pid)
        for r in d.diseases.values():
            assert r.fully_favourable == (r.satisfied == r.total)
        assert ("HighRiskPlant" in d.types) == any(r.fully_favourable for r in d.diseases.values())


def test_tanpa_level2_tidak_ada_fakta_tingkat2():
    d = diagnose_one(dict(SoilMoisture=0.5, SoilPH=6.8, SoilTemperature=27.0,
                          RelativeHumidity=0.6, Temperature=25.0), True, level2=False)
    assert not any(r.fully_favourable for r in d.diseases.values())


def test_tanaman_terisolasi_satu_sama_lain():
    kb = CassavaKB()
    kb.add_scenario(Scenario(1, False, {"SoilTemperature": 27.0}))
    kb.add_scenario(Scenario(2, False, {"SoilTemperature": 5.0}))
    kb.reason()
    assert "CBBSoilTemperature" in kb.diagnose(1).fired_conditions()
    assert kb.diagnose(2).fired_conditions() == []
