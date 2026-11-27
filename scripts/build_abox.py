"""Membangun A-Box demo + ekspor Turtle, dan menyimpan aturan SWRL sebagai OWL.

Keluaran (semantic-model/ontology_service/):
  abox/cassava-abox.owl            A-Box (RDF/XML): Field/Soil/Plant + sosa:Observation
  abox/cassava-abox.ttl            A-Box (Turtle)
  abox/cassava-abox-inferred.ttl   A-Box + fakta hasil inferensi Pellet (Turtle)
  owl/cassava-disease.ttl          T-Box asli paper dalam Turtle (hanya konversi format)
  owl/cassava-rules.owl            14 aturan SWRL tingkat-1 + 4 aturan tingkat-2 (RDF/XML)

A-Box berisi 5 tanaman demo (satu per kelas pada Eq. (1) paper) dengan data
sensor SIMULASI dari Tabel 1 paper, seed tetap sehingga dapat direproduksi.
Untuk kelas penyakit, sampel diundi ulang sampai semua kondisi penyakit itu
terpenuhi (skenario "favourable"); kelas Healthy diundi dari rentang Healthy.
Jalankan:  python scripts/build_abox.py
"""
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rdflib  # noqa: E402

from cassava_es.kb import ABOX_DIR, ABOX_IRI, EXT_IRI, OWL_DIR, TBOX_FILE, CassavaKB, Scenario  # noqa: E402
from cassava_es.rules import CLASSES  # noqa: E402
from cassava_es.simulate import simulate  # noqa: E402

SEED = 2023  # tahun paper; nilai bebas, hanya agar reprodusibel


def main() -> None:
    rng = random.Random(SEED)
    kb = CassavaKB(level2=True)
    manifest = []
    for pid, label in enumerate(CLASSES, start=1):
        s = simulate(label, rng, favourable=True)
        kb.add_scenario(Scenario(pid, s.rainy_season, s.values))
        manifest.append((pid, label, s.rainy_season, s.values))

    ABOX_DIR.mkdir(parents=True, exist_ok=True)
    # 1) A-Box sebelum inferensi
    kb.abox.save(file=str(ABOX_DIR / "cassava-abox.owl"), format="rdfxml")
    # 2) aturan SWRL sebagai artefak
    kb.rules_onto.save(file=str(OWL_DIR / "cassava-rules.owl"), format="rdfxml")

    g = rdflib.Graph()
    g.parse(ABOX_DIR / "cassava-abox.owl", format="xml")
    g.serialize(ABOX_DIR / "cassava-abox.ttl", format="turtle")

    # 3) jalankan reasoner; A-Box + fakta baru hasil inferensi -> Turtle
    secs = kb.reason()
    gi = rdflib.Graph()
    gi.parse(ABOX_DIR / "cassava-abox.owl", format="xml")
    TB = rdflib.Namespace("https://www.sti-innsbruck.at/cassava-disease#")
    AB = rdflib.Namespace(ABOX_IRI)
    EX = rdflib.Namespace(EXT_IRI)
    for pid, *_ in manifest:
        for s_, p_, o_ in kb.derived_triples(pid):
            subj = AB[s_]
            if p_ == "hasCondition":
                gi.add((subj, TB.hasCondition, TB[o_]))
                gi.add((TB[o_], TB.isConditionOf, subj))
            elif p_ == "isFullyFavourableFor":
                gi.add((subj, EX.isFullyFavourableFor, TB[o_]))
            elif p_ == "rdf:type":
                gi.add((subj, rdflib.RDF.type, EX[o_]))
    gi.bind("cd", TB); gi.bind("abox", AB); gi.bind("ext", EX)
    gi.serialize(ABOX_DIR / "cassava-abox-inferred.ttl", format="turtle")

    # 4) T-Box asli -> Turtle (hanya konversi serialisasi)
    gt = rdflib.Graph()
    gt.parse(TBOX_FILE, format="xml")
    gt.serialize(OWL_DIR / "cassava-disease.ttl", format="turtle")

    print(f"Pellet: {secs:.2f} s; triple A-Box={len(g)}, A-Box+inferensi={len(gi)}, T-Box={len(gt)}")
    for pid, label, rainy, vals in manifest:
        d = kb.diagnose(pid)
        vec = {k: round(v, 3) for k, v in d.semantic_vector().items()}
        print(pid, label, "rainy=", rainy, vals, "->", vec, "konsisten=", d.consistent)


if __name__ == "__main__":
    main()
