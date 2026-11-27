"""Pemeriksaan berkas keluaran: T-Box tak berubah, TTL setara OWL, A-Box konsisten dengan inferensi."""
import hashlib

import rdflib
from rdflib.compare import isomorphic

from cassava_es import cq
from cassava_es.kb import ABOX_DIR, ABOX_IRI, OWL_DIR, TBOX_FILE, CassavaKB, Scenario
from cassava_es.rules import RULES
from conftest import needs_java

# SHA-256 berkas cassava-disease.owl dari repositori asli (diukur saat berkas diterima).
ORIGINAL_TBOX_SHA256 = "598189aaedc6e8df9f471375872936b38a73a2afc9a27d8669bbe29001da8054"
TB = rdflib.Namespace("https://www.sti-innsbruck.at/cassava-disease#")
SOSA = rdflib.Namespace("http://www.w3.org/ns/sosa/")


def test_tbox_tidak_diubah():
    assert hashlib.sha256(TBOX_FILE.read_bytes()).hexdigest() == ORIGINAL_TBOX_SHA256


def test_tbox_ttl_setara_owl():
    a, b = rdflib.Graph(), rdflib.Graph()
    a.parse(OWL_DIR / "cassava-disease.owl", format="xml")
    b.parse(OWL_DIR / "cassava-disease.ttl", format="turtle")
    assert isomorphic(a, b)


def test_abox_ttl_setara_owl_dan_isi():
    a, b = rdflib.Graph(), rdflib.Graph()
    a.parse(ABOX_DIR / "cassava-abox.owl", format="xml")
    b.parse(ABOX_DIR / "cassava-abox.ttl", format="turtle")
    assert isomorphic(a, b)
    plants = set(b.subjects(rdflib.RDF.type, TB.Plant))
    assert len(plants) == 5
    assert len(set(b.subjects(rdflib.RDF.type, SOSA.Observation))) == 5 * 5
    assert len(set(b.subjects(rdflib.RDF.type, TB.Field))) == 5
    assert len(set(b.subjects(rdflib.RDF.type, TB.Soil))) == 5
    # A-Box tidak boleh memuat fakta turunan; T-Box tidak boleh memuat individu A-Box
    assert not list(b.triples((None, TB.hasCondition, None)))


def _scenarios_from_abox():
    g = rdflib.Graph()
    g.parse(ABOX_DIR / "cassava-abox.ttl", format="turtle")
    out = []
    for plant in sorted(g.subjects(rdflib.RDF.type, TB.Plant)):
        pid = int(str(plant).rsplit("_", 1)[1])
        soil = g.value(plant, TB.isPlantedIn)
        field = g.value(soil, TB.isPartOfField)
        rainy = g.value(field, TB.hasRainySeason).toPython()
        vals = {}
        for o in g.subjects(rdflib.RDF.type, SOSA.Observation):
            if g.value(o, SOSA.hasFeatureOfInterest) in (soil, field):
                vals[str(g.value(o, SOSA.observedProperty)).rsplit("#", 1)[1]] = float(g.value(o, SOSA.hasSimpleResult))
        out.append(Scenario(pid, rainy, vals))
    return out


@needs_java
def test_abox_inferred_sama_dengan_inferensi_ulang():
    kb = CassavaKB()
    for sc in _scenarios_from_abox():
        kb.add_scenario(sc)
    kb.reason()
    gi = rdflib.Graph()
    gi.parse(ABOX_DIR / "cassava-abox-inferred.ttl", format="turtle")
    for pid in range(1, 6):
        d = kb.diagnose(pid)
        assert d.consistent
        plant = rdflib.URIRef(f"{ABOX_IRI}plant_{pid}")
        in_file = {str(o).rsplit("#", 1)[1] for o in gi.objects(plant, TB.hasCondition)}
        assert in_file == set(d.fired_conditions()), pid


@needs_java
def test_berkas_aturan_swrl_menghasilkan_inferensi_sama():
    """cassava-rules.owl (artefak) dimuat ke dunia baru -> hasil sama dengan aturan di kode."""
    from owlready2 import World, sync_reasoner_pellet
    from cassava_es.kb import EXT_FILE, SOSA_FILE, _uri

    scs = _scenarios_from_abox()
    kb_ref = CassavaKB()
    for sc in scs:
        kb_ref.add_scenario(sc)
    kb_ref.reason()

    w = World()
    w.get_ontology(_uri(SOSA_FILE)).load()
    w.get_ontology(_uri(TBOX_FILE)).load()
    w.get_ontology(_uri(EXT_FILE)).load()
    w.get_ontology(_uri(OWL_DIR / "cassava-rules.owl")).load()
    w.get_ontology(_uri(ABOX_DIR / "cassava-abox.owl")).load()
    assert len(list(w.rules())) == 14 + 4
    sync_reasoner_pellet(w, infer_property_values=True, infer_data_property_values=True, debug=0)
    for sc in scs:
        plant = w[f"{ABOX_IRI}plant_{sc.plant_id}"]
        assert {c.name for c in plant.hasCondition} == set(kb_ref.diagnose(sc.plant_id).fired_conditions())


def test_cq():
    assert cq.cq1() == ["CassavaBacterialBlight", "CassavaBrownStreakDisease", "CassavaGreenMottle", "CassavaMosaicDisease"]
    assert [d for d, _ in cq.cq2("mosaic")] == ["CassavaGreenMottle", "CassavaMosaicDisease"]
    assert cq.cq2("kata-yang-tidak-ada") == []
    assert len(cq.cq3()) == 5 and len(cq.cq3_observations()) == 25
    assert "wilting" in cq.cq4("CassavaBacterialBlight")[0]
    assert cq.cq5("Viral") == ["CassavaBrownStreakDisease", "CassavaGreenMottle", "CassavaMosaicDisease"]
    assert cq.cq5("Bacterial") == ["CassavaBacterialBlight"]
    assert cq.cq5("Fungal") == []
