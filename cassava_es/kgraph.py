"""Representasi knowledge graph (T-Box dan A-Box) untuk ditampilkan di aplikasi.

T-Box : skema ontologi paper (cassava-disease.ttl) + ekstensi implementasi ini
        (cassava-extension.owl): kelas, hierarki ``subClassOf``, object property
        (domain -> range), serta individu T-Box (penyakit, kondisi, gejala,
        properti teramati) bila diminta.
A-Box : fakta untuk satu tanaman yang didiagnosa: Plant/Soil/Field, observasi
        sensor (SOSA), dan fakta BARU hasil inferensi Pellet (``hasCondition``,
        ``isFullyFavourableFor``, tipe ``HighRiskPlant``).

Modul ini hanya membaca file ontologi (rdflib) dan data hasil diagnosa, tanpa
menjalankan reasoner, sehingga murah dipanggil ulang.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

from .kb import EXT_FILE, FIELD_PROPS, OWL_DIR, SOIL_PROPS

ASSERTED, INFERRED, SCHEMA = "asserted", "inferred", "schema"


def frag(uri) -> str:
    s = str(uri)
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


@dataclass
class KGraph:
    nodes: dict = field(default_factory=dict)  # id -> {"label", "category", "desc"}
    links: list = field(default_factory=list)  # {"source", "target", "label", "kind"}

    def node(self, nid: str, category: str, label: str | None = None, desc: str = "") -> str:
        if nid not in self.nodes:
            self.nodes[nid] = {"label": label or nid, "category": category, "desc": desc}
        return nid

    def link(self, s: str, t: str, label: str, kind: str = ASSERTED) -> None:
        if not any(x["source"] == s and x["target"] == t and x["label"] == label for x in self.links):
            self.links.append({"source": s, "target": t, "label": label, "kind": kind})

    def triples(self) -> list[dict]:
        return [{"Subjek": self.nodes[x["source"]]["label"], "Predikat": x["label"],
                 "Objek": self.nodes[x["target"]]["label"], "Jenis": x["kind"]} for x in self.links]


# --------------------------------------------------------------------------- T-Box
TBOX_CATEGORIES = ("Kelas", "Kelas SOSA", "Kelas ekstensi", "Penyakit", "Kondisi", "Gejala",
                   "Properti teramati")
_IND_CATEGORY = {"BacterialPlantDisease": "Penyakit", "ViralPlantDisease": "Penyakit",
                 "FungalPlantDisease": "Penyakit", "DiseaseSymptom": "Gejala",
                 "ObservableProperty": "Properti teramati"}


@lru_cache(maxsize=1)
def _tbox_rdf() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(OWL_DIR / "cassava-disease.ttl", format="turtle")
    ext = rdflib.Graph()
    ext.parse(EXT_FILE, format="xml")
    return g, ext


def _class_category(uri, ext_classes: set) -> str:
    if "sosa" in str(uri):
        return "Kelas SOSA"
    return "Kelas ekstensi" if uri in ext_classes else "Kelas"


def tbox_graph(include_individuals: bool = False) -> KGraph:
    g, ext = _tbox_rdf()
    kg = KGraph()
    ext_classes = {c for c in ext.subjects(RDF.type, OWL.Class) if isinstance(c, rdflib.URIRef)}
    both = g + ext

    def cls(u) -> str:
        return kg.node(frag(u), _class_category(u, ext_classes), desc="owl:Class")

    for c in both.subjects(RDF.type, OWL.Class):
        if isinstance(c, rdflib.URIRef):
            cls(c)
    for a, b in both.subject_objects(RDFS.subClassOf):
        if isinstance(a, rdflib.URIRef) and isinstance(b, rdflib.URIRef):
            kg.link(cls(a), cls(b), "subClassOf", SCHEMA)
    for p in both.subjects(RDF.type, OWL.ObjectProperty):
        for d in both.objects(p, RDFS.domain):
            for r in both.objects(p, RDFS.range):
                if isinstance(d, rdflib.URIRef) and isinstance(r, rdflib.URIRef):
                    kg.link(cls(d), cls(r), frag(p), SCHEMA)

    if include_individuals:
        for i in sorted(set(g.subjects(RDF.type, OWL.NamedIndividual))):
            types = [t for t in g.objects(i, RDF.type) if t != OWL.NamedIndividual]
            if not types:
                continue
            tname = frag(types[0])
            cat = _IND_CATEGORY.get(tname, "Kondisi" if tname.endswith("Condition") else "Penyakit")
            nid = kg.node(frag(i), cat, desc=f"individu {tname}")
            kg.link(nid, cls(types[0]), "rdf:type", SCHEMA)
            for p, o in g.predicate_objects(i):
                if p != RDF.type and isinstance(o, rdflib.URIRef) and (o, RDF.type, OWL.NamedIndividual) in g:
                    kg.link(nid, frag(o), frag(p), SCHEMA)
    return kg


def tbox_summary() -> dict:
    g, ext = _tbox_rdf()
    both = g + ext
    count = lambda t: len({s for s in both.subjects(RDF.type, t) if isinstance(s, rdflib.URIRef)})  # noqa: E731
    return {"Kelas": count(OWL.Class), "Object property": count(OWL.ObjectProperty),
            "Datatype property": count(OWL.DatatypeProperty), "Individu": count(OWL.NamedIndividual)}


# --------------------------------------------------------------------------- A-Box
ABOX_CATEGORIES = ("Tanaman / tanah / ladang", "Observasi", "Nilai", "Properti teramati",
                   "Kondisi (inferensi)", "Penyakit", "Kelas")


def _disease_of(condition: str) -> str | None:
    g, _ = _tbox_rdf()
    for o in g.objects(rdflib.URIRef("https://www.sti-innsbruck.at/cassava-disease#" + condition),
                       rdflib.URIRef("https://www.sti-innsbruck.at/cassava-disease#enablesDisease")):
        return frag(o)
    return None


def abox_graph(values: Mapping[str, float] | None, rainy: bool,
               derived: Iterable[tuple[str, str, str]] = (), plant_id: int = 1,
               show_types: bool = True) -> KGraph:
    """A-Box satu tanaman: fakta masukan (asserted) + fakta baru Pellet (inferred)."""
    kg = KGraph()
    feat = "Tanaman / tanah / ladang"
    plant = kg.node(f"plant_{plant_id}", feat, desc="individu Plant")
    soil = kg.node(f"soil_{plant_id}", feat, desc="individu Soil")
    fld = kg.node(f"field_{plant_id}", feat, desc="individu Field")
    kg.link(plant, soil, "isPlantedIn")
    kg.link(soil, fld, "isPartOfField")
    lit = kg.node(f"lit:rainy_{plant_id}", "Nilai", label=str(bool(rainy)).lower(), desc="xsd:boolean")
    kg.link(fld, lit, "hasRainySeason")
    if show_types:
        for ind, c in ((plant, "Plant"), (soil, "Soil"), (fld, "Field")):
            kg.link(ind, kg.node(c, "Kelas", desc="owl:Class (T-Box)"), "rdf:type")

    for prop, val in (values or {}).items():
        if val is None:
            continue
        foi, tag = (soil, f"soil_{plant_id}") if prop in SOIL_PROPS else (fld, f"field_{plant_id}")
        if prop not in SOIL_PROPS and prop not in FIELD_PROPS:
            continue
        obs = kg.node(f"obs_{tag}_{prop}", "Observasi", desc="sosa:Observation")
        kg.link(obs, kg.node(prop, "Properti teramati", desc="sosa:ObservableProperty (T-Box)"),
                "observedProperty")
        kg.link(obs, foi, "hasFeatureOfInterest")
        kg.link(obs, kg.node(f"lit:{obs}", "Nilai", label=f"{float(val):g}", desc="xsd:double"),
                "hasSimpleResult")
        if show_types:
            kg.link(obs, kg.node("Observation", "Kelas", desc="sosa:Observation"), "rdf:type")

    for s, p, o in derived:
        if p == "hasCondition":
            c = kg.node(o, "Kondisi (inferensi)", desc="DiseaseCondition (T-Box), dipicu aturan SWRL")
            kg.link(plant, c, "hasCondition", INFERRED)
            d = _disease_of(o)
            if d:
                kg.link(c, kg.node(d, "Penyakit", desc="individu PlantDisease (T-Box)"), "enablesDisease",
                        SCHEMA)
        elif p == "isFullyFavourableFor":
            kg.link(plant, kg.node(o, "Penyakit", desc="individu PlantDisease (T-Box)"),
                    "isFullyFavourableFor", INFERRED)
        elif p == "rdf:type":
            kg.link(plant, kg.node(o, "Kelas", desc="kelas hasil klasifikasi DL"), "rdf:type", INFERRED)
    return kg


# --------------------------------------------------------------------------- ECharts
PALETTE = {
    "Kelas": "#0F7A43", "Kelas SOSA": "#0E7490", "Kelas ekstensi": "#7C3AED", "Penyakit": "#C0392B",
    "Kondisi": "#D97706", "Gejala": "#8D5524", "Properti teramati": "#2563EB",
    "Tanaman / tanah / ladang": "#0F7A43", "Observasi": "#0E7490", "Nilai": "#64748B",
    "Kondisi (inferensi)": "#D97706",
}
EDGE = {ASSERTED: "#94A39A", SCHEMA: "#94A39A", INFERRED: "#E11D48"}


def to_echarts(kg: KGraph, categories: Iterable[str], height_hint: int = 560) -> dict:
    cats = [c for c in categories if any(n["category"] == c for n in kg.nodes.values())]
    idx = {c: i for i, c in enumerate(cats)}
    degree: dict[str, int] = {}
    for x in kg.links:
        degree[x["source"]] = degree.get(x["source"], 0) + 1
        degree[x["target"]] = degree.get(x["target"], 0) + 1
    nodes = []
    for nid, n in kg.nodes.items():
        is_lit = n["category"] == "Nilai"
        nodes.append({
            "id": nid, "name": n["label"], "category": idx[n["category"]],
            "value": n["desc"], "symbol": "roundRect" if is_lit else "circle",
            "symbolSize": 16 if is_lit else min(46, 18 + 3 * degree.get(nid, 0)),
            "label": {"show": True, "position": "right", "fontSize": 11},
        })
    links = []
    for x in kg.links:
        inferred = x["kind"] == INFERRED
        links.append({
            "source": x["source"], "target": x["target"], "value": x["label"],
            "lineStyle": {"color": EDGE[x["kind"]], "width": 2.4 if inferred else 1.2,
                          "type": "dashed" if inferred else "solid", "curveness": 0.08, "opacity": 0.9},
            "label": {"show": True, "formatter": x["label"], "fontSize": 9,
                      "color": EDGE[x["kind"]] if inferred else "#6B7A70"},
        })
    return {
        "tooltip": {"trigger": "item"},
        "legend": [{"data": cats, "top": 0, "type": "scroll", "textStyle": {"fontSize": 11}}],
        "color": [PALETTE.get(c, "#64748B") for c in cats],
        "series": [{
            "type": "graph", "layout": "force", "roam": True, "draggable": True,
            "top": 40, "bottom": 10, "left": 10, "right": 10,
            "categories": [{"name": c} for c in cats],
            "data": nodes, "links": links,
            "edgeSymbol": ["none", "arrow"], "edgeSymbolSize": 7,
            "force": {"repulsion": 260, "edgeLength": [70, 140], "gravity": 0.08},
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 3}},
            "labelLayout": {"hideOverlap": True},
        }],
    }
