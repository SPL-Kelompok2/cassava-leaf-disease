import pytest

from cassava_es.decision import combine
from cassava_es.rules import CLASSES

# Contoh numerik Bagian 3.2.3 paper (urutan kelas: CBB, CBSD, CGM, CMD, Healthy)
P_VIS = dict(zip(CLASSES, [0.21, 0.05, 0.05, 0.19, 0.5]))
P_SEM = dict(zip(CLASSES, [0.49, 0.3, 0.4, 0.5, 0.19]))


def test_hanya_visi_dan_hanya_semantik_sesuai_paper():
    assert combine(P_VIS, None)[1] == "Healthy"  # semantik=P_VIS dipakai sebagai satu-satunya
    assert combine(P_SEM, None)[1] == "CMD"


def test_bobot_03_08_sesuai_paper():
    probs, label = combine(P_SEM, P_VIS, w_semantic=0.8, w_vision=0.3)
    expected = [0.414, 0.232, 0.305, 0.415, 0.275]
    for c, e in zip(CLASSES, expected):
        assert probs[c] == pytest.approx(e, abs=0.0006), c
    assert label == "CMD"


def test_bobot_sama_sesuai_paper():
    probs, label = combine(P_SEM, P_VIS, 0.5, 0.5)
    expected = [0.35, 0.175, 0.225, 0.345, 0.345]
    for c, e in zip(CLASSES, expected):
        assert probs[c] == pytest.approx(e, abs=1e-9), c
    assert label == "CBB"


def test_seri_memilih_kelas_pertama():
    sem = dict.fromkeys(CLASSES, 0.2)
    assert combine(sem)[1] == "CBB"


def test_bobot_tidak_valid():
    with pytest.raises(ValueError):
        combine(P_SEM, P_VIS, -1, 1)
    with pytest.raises(ValueError):
        combine(P_SEM, P_VIS, 0, 0)
