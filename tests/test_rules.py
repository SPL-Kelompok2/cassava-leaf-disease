from collections import Counter

from cassava_es.rules import DISEASES, RULES, RULES_BY_ID, rules_for
from cassava_es.simulate import TABLE1


def test_jumlah_aturan():
    assert len(RULES) == 14
    assert Counter(r.disease for r in RULES) == {"CBB": 4, "CBSD": 3, "CGM": 3, "CMD": 4}
    assert [r.rule_id for r in RULES] == [f"R{i:02d}" for i in range(1, 15)]


def test_ambang_sama_dengan_tabel1_paper():
    """Lower/Upper aturan harus sama dengan kolom Lower/Upper Tabel 1 (13 baris)."""
    n = 0
    for r in RULES:
        if r.observed_property is None:
            continue  # musim hujan: bukan baris Tabel 1
        lower, upper, *_ = TABLE1[r.disease][r.observed_property]
        assert r.lower == lower, r.rule_id
        # Tabel 1 mencatat batas atas fisik (1, 40, 50) pada baris 'tanpa batas atas'
        # di aturan; hanya baris dengan batas atas aturan yang harus sama.
        if r.upper is not None:
            assert r.upper == upper, r.rule_id
        n += 1
    assert n == 13


def test_kondisi_ada_di_tbox():
    from cassava_es.kb import CassavaKB
    kb = CassavaKB()
    for r in RULES:
        ind = kb.tbox[r.condition]
        assert ind is not None, r.condition
        assert ind.enablesDisease.name.startswith("Cassava")
        assert kb.tbox.DiseaseCondition in ind.is_a or any(
            issubclass(c, kb.tbox.DiseaseCondition) for c in ind.is_a)
    assert len(kb.tbox.DiseaseCondition.instances()) == 14


def test_swrl_contoh():
    assert RULES_BY_ID["R04"].swrl() == (
        "Plant(?plant), isPlantedIn(?plant, ?soil), isFeatureOfInterestOf(?soil, ?obs), "
        "observedProperty(?obs, SoilTemperature), hasSimpleResult(?obs, ?result), "
        "greaterThanOrEqual(?result, 25), lessThanOrEqual(?result, 30) -> hasCondition(?plant, CBBSoilTemperature)")
    assert "isPartOfField(?soil, ?field), isFeatureOfInterestOf(?field, ?obs)" in RULES_BY_ID["R05"].swrl()


def test_python_evaluator_batas_inklusif():
    r = RULES_BY_ID["R03"]
    assert r.satisfied_by(6.5) and r.satisfied_by(7.2)
    assert not r.satisfied_by(6.4999) and not r.satisfied_by(7.2001) and not r.satisfied_by(None)
