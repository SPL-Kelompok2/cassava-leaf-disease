import pytest

from cassava_es.disease_info import DISEASE_INFO, GENERAL_DISCLAIMER, get_info
from cassava_es.rules import CLASSES, DISEASE_LABEL


def test_semua_kelas_punya_info():
    assert set(DISEASE_INFO) == set(CLASSES)
    for c in CLASSES:
        assert get_info(c).code == c


def test_kolom_wajib_terisi():
    for c, d in DISEASE_INFO.items():
        for fld in ("name_en", "name_id", "category", "agent", "definition", "impact", "favourable"):
            assert getattr(d, fld).strip(), (c, fld)
        assert d.symptoms and d.management and d.sources, c
        assert d.name_en.split()[0] in DISEASE_LABEL[c] or c == "Healthy"


def test_penyakit_punya_cara_penyebaran_dan_sumber_berurl():
    for c in ("CBB", "CBSD", "CGM", "CMD"):
        assert DISEASE_INFO[c].spread, c
    for d in DISEASE_INFO.values():
        for title, url in d.sources:
            assert title and url.startswith("https://"), (d.code, url)


def test_cgm_adalah_tungau_bukan_virus():
    d = get_info("CGM")
    assert "tungau" in d.agent.lower() and "Mononychellus" in d.agent
    assert "Hama" in d.category
    joined = " ".join(d.notes).lower()
    assert "mottle" in joined and "green mite" in joined  # perbedaan penamaan dijelaskan


def test_agen_penyebab_sesuai_sumber():
    assert "Xanthomonas" in get_info("CBB").agent
    assert "Ipomovirus" in get_info("CBSD").agent
    assert "begomovirus" in get_info("CMD").agent.lower()
    assert "Bemisia" in " ".join(get_info("CMD").spread) and "Bemisia" in " ".join(get_info("CBSD").spread)


def test_tidak_ada_dosis_atau_merek_pestisida():
    banned = ("ml/", "gram/", "g/l", "dosis", "insektisida", "akarisida", "bakterisida")
    for d in DISEASE_INFO.values():
        text = " ".join([d.definition, d.impact, *d.management, *d.notes]).lower()
        assert not any(b in text for b in banned), d.code


def test_healthy_menyebut_keterbatasan_dan_disclaimer_ada():
    assert any("bukan jaminan" in n for n in get_info("Healthy").notes)
    assert "bukan diagnosis" in GENERAL_DISCLAIMER


def test_kelas_tak_dikenal():
    with pytest.raises(KeyError):
        get_info("XYZ")
