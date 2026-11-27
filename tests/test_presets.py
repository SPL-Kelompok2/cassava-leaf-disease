from cassava_es.decision import combine
from cassava_es.kb import diagnose_one
from cassava_es.presets import EXPECTED_TOP, PRESETS
from conftest import needs_java

pytestmark = needs_java


def test_setiap_contoh_menghasilkan_kelas_teratas_unik_yang_dinamai():
    for name, expected in EXPECTED_TOP.items():
        p = dict(PRESETS[name])
        rainy = p.pop("rainy")
        d = diagnose_one(p, rainy)
        probs, label = combine(d.semantic_vector())
        assert label == expected, (name, probs)
        others = [v for c, v in probs.items() if c != expected]
        assert probs[expected] > max(others), (name, probs)
