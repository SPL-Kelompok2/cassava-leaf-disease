"""Competency questions (CQ) ontologi paper, dijawab dengan SPARQL (rdflib).

Lima CQ berikut disalin dari Bagian 3.1.1 paper:
  CQ1 Penyakit apa saja yang mungkin menyerang tanaman singkong?
  CQ2 Penyakit apa yang mungkin ada bila gejalanya A, B, ...?
  CQ3 Bagaimana lingkungan tanaman dan observasi sensor terkait dengan tanaman?
  CQ4 Apa gejala tanaman singkong dengan penyakit X?
  CQ5 Penyakit apa yang disebabkan virus (atau bakteri, atau jamur)?

Catatan: gejala pada T-Box berupa satu individu DiseaseSymptom per penyakit dengan teks
bebas (skos:definition); CQ2 dijawab dengan pencarian kata kunci pada teks gejala.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import rdflib

from .kb import ABOX_DIR, OWL_DIR

PREFIXES = """
PREFIX cd:   <https://www.sti-innsbruck.at/cassava-disease#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
"""


@lru_cache(maxsize=1)
def graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(Path(OWL_DIR) / "cassava-disease.ttl", format="turtle")
    g.parse(Path(ABOX_DIR) / "cassava-abox-inferred.ttl", format="turtle")
    return g


def _run(q: str, **bind) -> list[tuple]:
    res = graph().query(PREFIXES + q, initBindings=bind or None)
    return [tuple(str(x) for x in row) for row in res]


def _frag(uri: str) -> str:
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def cq1() -> list[str]:
    rows = _run("""
        SELECT DISTINCT ?d WHERE {
          ?d a ?c . ?c rdfs:subClassOf* cd:PlantDisease .
          ?d a owl:NamedIndividual .
        } ORDER BY ?d""")
    return [_frag(r[0]) for r in rows]


def cq2(keyword: str) -> list[tuple[str, str]]:
    rows = _run("""
        SELECT ?d ?def WHERE {
          ?s a cd:DiseaseSymptom ; cd:isCausedByDisease ?d ; skos:definition ?def .
          FILTER(CONTAINS(LCASE(STR(?def)), LCASE(?kw)))
        } ORDER BY ?d""", kw=rdflib.Literal(keyword))
    return [(_frag(d), t) for d, t in rows]


def cq3() -> list[tuple[str, ...]]:
    rows = _run("""
        SELECT ?plant ?soil ?field ?rainy WHERE {
          ?plant a cd:Plant ; cd:isPlantedIn ?soil .
          ?soil cd:isPartOfField ?field . ?field cd:hasRainySeason ?rainy .
        } ORDER BY ?plant""")
    return [tuple(_frag(x) for x in r[:3]) + (r[3],) for r in rows]


def cq3_observations() -> list[tuple[str, str, str]]:
    rows = _run("""
        SELECT ?foi ?prop ?val WHERE {
          ?o a sosa:Observation ; sosa:hasFeatureOfInterest ?foi ;
             sosa:observedProperty ?prop ; sosa:hasSimpleResult ?val .
        } ORDER BY ?foi ?prop""")
    return [(_frag(a), _frag(b), c) for a, b, c in rows]


def cq4(disease_individual: str) -> list[str]:
    rows = _run("""
        SELECT ?def WHERE {
          ?s a cd:DiseaseSymptom ; cd:isCausedByDisease ?d ; skos:definition ?def .
        }""", d=rdflib.URIRef("https://www.sti-innsbruck.at/cassava-disease#" + disease_individual))
    return [r[0] for r in rows]


def cq5(kind: str) -> list[str]:
    """kind: 'Viral' | 'Bacterial' | 'Fungal'."""
    cls = rdflib.URIRef(f"https://www.sti-innsbruck.at/cassava-disease#{kind}PlantDisease")
    rows = _run("SELECT ?d WHERE { ?d a ?cls } ORDER BY ?d", cls=cls)
    return [_frag(r[0]) for r in rows]
