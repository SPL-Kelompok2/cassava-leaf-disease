import json

import pytest

from cassava_es.decision import combine
from cassava_es.engine import DecisionEngine, FinalResult
from cassava_es.kb import Diagnosis, DiseaseResult
from cassava_es.presets import PRESETS
from cassava_es.rules import CLASSES, DISEASE_LABEL
from conftest import needs_java

# Contoh numerik Bagian 3.2.3 paper (urutan CBB, CBSD, CGM, CMD, Healthy)
P_VIS = dict(zip(CLASSES, [0.21, 0.05, 0.05, 0.19, 0.5]))
P_SEM = dict(zip(CLASSES, [0.49, 0.3, 0.4, 0.5, 0.19]))


def fake_diagnosis(vec):
    """Diagnosis buatan tangan: probabilitas semantik = vec (tanpa Pellet)."""
    diseases = {
        c: DiseaseResult(c, 0, 1, vec[c], f"gejala {c}", [f"penjelasan {c} 1", f"penjelasan {c} 2"], [], False)
        for c in CLASSES[:4]
    }
    return Diagnosis(1, diseases, vec["Healthy"], [])


def engine():
    return DecisionEngine(semantic_fn=lambda *a, **k: fake_diagnosis(P_SEM))


# ---------------------------------------------------------------- fusi (Eq. 1)
def test_fusi_bobot_sama_cocok_contoh_paper():
    r = engine().predict(sensors={"SoilPH": 6.0}, vision_vector=P_VIS, w_semantic=0.5, w_vision=0.5)
    assert r.mode == "fusion" and r.label == "CBB"
    expected = [0.35, 0.175, 0.225, 0.345, 0.345]
    for c, e in zip(CLASSES, expected):
        assert r.probabilities[c] == pytest.approx(e, abs=1e-9)


def test_fusi_bobot_semantik_besar_cocok_contoh_paper():
    r = engine().predict(sensors={"x": 1}, vision_vector=P_VIS, w_semantic=0.8, w_vision=0.3)
    assert r.label == "CMD"
    for c, e in zip(CLASSES, [0.414, 0.232, 0.305, 0.415, 0.275]):
        assert r.probabilities[c] == pytest.approx(e, abs=6e-4)


def test_engine_konsisten_dengan_combine():
    r = engine().predict(sensors={"x": 1}, vision_vector=P_VIS, w_semantic=0.3, w_vision=0.7)
    probs, label = combine(P_SEM, P_VIS, 0.3, 0.7)
    assert r.label == label and r.probabilities == probs


def test_kepastian_visual_dan_pengetahuan():
    r = engine().predict(sensors={"x": 1}, vision_vector=P_VIS, w_semantic=0.8, w_vision=0.3)
    assert r.visual_certainty == pytest.approx(P_VIS[r.label])
    assert r.knowledge_certainty == pytest.approx(P_SEM[r.label])


# ---------------------------------------------------------------- mode lain
def test_semantik_saja_sama_dengan_perilaku_lama():
    r = engine().predict(sensors={"x": 1})
    assert r.mode == "semantic_only" and r.label == "CMD"  # argmax P_SEM
    assert r.vision is None and r.visual_certainty is None and not r.visual_supports
    assert r.weights == {"semantic": 1.0, "vision": 0.0}


def test_visi_saja_tanpa_penjelasan_ontologi():
    r = engine().predict(vision_vector=P_VIS)
    assert r.mode == "vision_only" and r.label == "Healthy"
    assert r.semantic is None and r.knowledge_certainty is None
    assert r.explanations == [] and r.symptoms == "" and r.diagnosis is None
    assert r.probabilities == pytest.approx(P_VIS)


def test_tanpa_masukan_apa_pun_error():
    with pytest.raises(ValueError):
        engine().predict()


def test_vektor_visi_tidak_lengkap_error():
    with pytest.raises(KeyError):
        engine().predict(vision_vector={"CBB": 1.0})


def test_citra_tanpa_model_visi_error_jelas():
    with pytest.raises(RuntimeError, match="model visi"):
        engine().predict(image=b"bukan-citra", sensors={"x": 1})


def test_bobot_tidak_valid_diteruskan_sebagai_error():
    with pytest.raises(ValueError):
        engine().predict(sensors={"x": 1}, vision_vector=P_VIS, w_semantic=0, w_vision=0)


def test_diagnosis_siap_pakai_tidak_menjalankan_semantik_lagi():
    def boom(*a, **k):
        raise AssertionError("tidak boleh dipanggil")

    r = DecisionEngine(semantic_fn=boom).predict(diagnosis=fake_diagnosis(P_SEM), vision_vector=P_VIS)
    assert r.mode == "fusion"


# ---------------------------------------------------------------- penjelasan & info
def test_penjelasan_dan_info_untuk_kelas_pemenang():
    r = engine().predict(sensors={"x": 1}, vision_vector=P_VIS, w_semantic=0.8, w_vision=0.3)
    assert r.label == "CMD" and r.explanations == ["penjelasan CMD 1", "penjelasan CMD 2"]
    assert r.symptoms == "gejala CMD"
    assert r.info.code == "CMD" and "mosaik" in r.info.definition.lower()
    assert r.name == DISEASE_LABEL["CMD"]


def test_healthy_tanpa_penjelasan_penyakit_tetapi_ada_info():
    vec = dict(zip(CLASSES, [0.0, 0.0, 0.0, 0.0, 1.0]))
    r = DecisionEngine(semantic_fn=lambda *a, **k: fake_diagnosis(vec)).predict(sensors={"x": 1})
    assert r.label == "Healthy" and r.explanations == [] and r.info.code == "Healthy"
    assert not r.visual_supports


def test_visual_supports_hanya_bila_p_visi_kelas_ambang_05():
    hi = dict(zip(CLASSES, [0.0, 0.0, 0.0, 0.9, 0.1]))
    lo = dict(zip(CLASSES, [0.2, 0.2, 0.2, 0.3, 0.1]))
    cmd_sem = dict(zip(CLASSES, [0.0, 0.0, 0.0, 1.0, 0.0]))
    e = DecisionEngine(semantic_fn=lambda *a, **k: fake_diagnosis(cmd_sem))
    assert e.predict(sensors={"x": 1}, vision_vector=hi).visual_supports is True
    assert e.predict(sensors={"x": 1}, vision_vector=lo).visual_supports is False


def test_seri_dilaporkan():
    flat = dict.fromkeys(CLASSES, 0.2)
    r = DecisionEngine(semantic_fn=lambda *a, **k: fake_diagnosis(flat)).predict(sensors={"x": 1})
    assert r.label == "CBB" and r.ties == ["CBSD", "CGM", "CMD", "Healthy"]


def test_json_mirip_gambar6_dan_dapat_diserialisasi():
    r = engine().predict(sensors={"x": 1}, vision_vector=P_VIS, w_semantic=0.8, w_vision=0.3)
    d = json.loads(r.to_json())
    for key in ("disease", "explanations", "visual_certainty", "knowledge_certainty",
                "disease_info", "probabilities", "weights", "disclaimer"):
        assert key in d
    assert d["disease"] == "CMD" and d["disease_info"]["code"] == "CMD"
    assert isinstance(d["disease_info"]["symptoms"], list)
    assert isinstance(r, FinalResult)


# ---------------------------------------------------------------- dengan Pellet asli
@needs_java
def test_engine_dengan_pellet_asli_cmd_dan_vision_mengubah_keputusan():
    p = dict(PRESETS["Contoh CMD"])
    rainy = p.pop("rainy")
    eng = DecisionEngine()
    sem_only = eng.predict(sensors=p, rainy=rainy)
    assert sem_only.label == "CMD" and sem_only.knowledge_certainty == pytest.approx(1.0)
    assert sem_only.explanations and sem_only.diagnosis.consistent
    # visi sangat yakin 'Healthy' + bobot sama -> keputusan berubah (semantik CMD=1.0, Healthy=1-9/14)
    vis = dict(zip(CLASSES, [0.0, 0.0, 0.0, 0.0, 1.0]))
    fused = eng.predict(diagnosis=sem_only.diagnosis, vision_vector=vis, w_semantic=0.5, w_vision=0.5)
    assert fused.label == "Healthy"
    d = json.loads(fused.to_json())
    assert d["disease_info"]["code"] == "Healthy"
