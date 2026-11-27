"""Knowledge base + mesin inferensi (owlready2 + Pellet).

Alur (mengikuti Bagian 3.2.2 paper):
  1. muat T-Box (cassava-disease.owl) dan SOSA;
  2. isi A-Box: Field / Soil / Plant + sosa:Observation dari sensor;
  3. pasang aturan SWRL (rules.py) dan jalankan reasoner Pellet;
  4. baca fakta baru ``hasCondition`` dan hitung vektor penyakit.

Perhitungan probabilitas semantik MENGIKUTI kode repositori asli (ontology.py):
    p(penyakit) = (#kondisi terpenuhi untuk penyakit) / (#kondisi penyakit tsb)
    p(sehat)    = 1 - (#seluruh kondisi terpenuhi) / (#seluruh kondisi = 14)
"""

from __future__ import annotations

import datetime as _dt
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

# owlready2 0.37 memunculkan SyntaxWarning (regex) pada Python >= 3.12; tidak berbahaya.
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from owlready2 import Imp, World, sync_reasoner_pellet  # noqa: E402

from .rules import (  # noqa: E402
    DISEASE_IRI_NAME,
    DISEASES,
    RULES,
    RULES_BY_CONDITION,
    Rule,
    rules_for,
)

ROOT = Path(__file__).resolve().parent.parent
OWL_DIR = ROOT / "semantic-model" / "ontology_service" / "owl"
ABOX_DIR = ROOT / "semantic-model" / "ontology_service" / "abox"

TBOX_FILE = OWL_DIR / "cassava-disease.owl"  # T-Box asli paper (tidak diubah)
SOSA_FILE = OWL_DIR / "sosa.owl"
EXT_FILE = OWL_DIR / "cassava-extension.owl"  # ekstensi T-Box (milik implementasi ini)

ABOX_IRI = "http://example.org/cassava-abox#"
RULES_IRI = "http://example.org/cassava-rules#"
EXT_IRI = "http://example.org/cassava-extension#"

# Nama sensor di aplikasi/tes = nama individu sosa:ObservableProperty pada T-Box.
SOIL_PROPS = ("SoilMoisture", "SoilPH", "SoilTemperature")
FIELD_PROPS = ("RelativeHumidity", "Temperature")


def _uri(p: Path) -> str:
    return "file://" + p.resolve().as_posix()


# --------------------------------------------------------------------------- hasil
@dataclass
class ConditionTrace:
    rule_id: str
    condition: str
    disease: str
    fired: bool  # menurut Pellet (hasCondition hasil inferensi)
    observed: object  # nilai yang diamati (float/bool) atau None
    if_text: str  # bagian IF dalam bahasa manusia
    swrl: str
    definition: str  # anotasi skos:definition pada individu kondisi (T-Box paper)
    source: str
    python_agrees: bool  # evaluasi independen Python == hasil Pellet


@dataclass
class DiseaseResult:
    code: str
    satisfied: int
    total: int
    probability: float
    symptoms: str
    explanations: list  # definisi kondisi yang terpenuhi (penjelasan gaya paper)
    traces: list  # ConditionTrace untuk SEMUA kondisi penyakit ini
    fully_favourable: bool  # fakta tingkat-2 (ekstensi)


@dataclass
class Diagnosis:
    plant_id: int
    diseases: dict  # kode -> DiseaseResult
    healthy_probability: float
    types: list  # kelas hasil klasifikasi DL (mis. HighRiskPlant)
    seconds: float = 0.0
    consistent: bool = True  # Pellet == evaluasi Python untuk semua aturan
    derived: list = field(default_factory=list)  # tripel fakta baru hasil inferensi

    def semantic_vector(self) -> dict:
        v = {c: r.probability for c, r in self.diseases.items()}
        v["Healthy"] = self.healthy_probability
        return v

    def fired_conditions(self) -> list:
        return [t.condition for r in self.diseases.values() for t in r.traces if t.fired]


@dataclass
class Scenario:
    plant_id: int
    rainy_season: bool
    values: Mapping[str, float]
    timestamp: _dt.datetime = field(default_factory=lambda: _dt.datetime(2026, 11, 25, 8, 0, 0))


# --------------------------------------------------------------------------- KB
class CassavaKB:
    """Satu 'dunia' owlready2 yang terisolasi (aman dipakai berulang oleh app)."""

    def __init__(self, level2: bool = True):
        self.level2 = level2
        self.world = World()
        self.sosa = self.world.get_ontology(_uri(SOSA_FILE)).load()
        self.tbox = self.world.get_ontology(_uri(TBOX_FILE)).load()
        self.ext = self.world.get_ontology(_uri(EXT_FILE)).load()
        self.abox = self.world.get_ontology(ABOX_IRI)
        self.abox.imported_ontologies.append(self.tbox)
        self.abox.imported_ontologies.append(self.ext)
        self.rules_onto = self.world.get_ontology(RULES_IRI)
        self.rules_onto.imported_ontologies.append(self.tbox)
        self.rules_onto.imported_ontologies.append(self.ext)
        self._plants: dict[int, Scenario] = {}
        self._install_rules()
        self.reasoned = False
        self.seconds = 0.0

    # -- aturan
    def _install_rules(self) -> None:
        ns = [self.tbox, self.sosa, self.ext]
        with self.rules_onto:
            for r in RULES:
                Imp().set_as_rule(r.swrl(), namespaces=ns)
            if self.level2:
                for d in DISEASES:
                    conds = [r.condition for r in rules_for(d)]
                    body = ", ".join(f"hasCondition(?plant, {c})" for c in conds)
                    Imp().set_as_rule(
                        f"Plant(?plant), {body} -> isFullyFavourableFor(?plant, {DISEASE_IRI_NAME[d]})",
                        namespaces=ns,
                    )

    # -- A-Box
    def add_scenario(self, sc: Scenario) -> None:
        t, s = self.tbox, self.sosa
        pid = sc.plant_id
        with self.abox:
            field_ = t.Field(f"field_{pid}", namespace=self.abox)
            field_.hasRainySeason = bool(sc.rainy_season)
            soil = t.Soil(f"soil_{pid}", namespace=self.abox)
            soil.isPartOfField = field_
            plant = t.Plant(f"plant_{pid}", namespace=self.abox)
            plant.isPlantedIn = soil
            for prop, val in sc.values.items():
                if val is None:
                    continue
                if prop in SOIL_PROPS:
                    foi, tag = soil, f"soil_{pid}"
                elif prop in FIELD_PROPS:
                    foi, tag = field_, f"field_{pid}"
                else:
                    raise KeyError(f"properti teramati tidak dikenal: {prop}")
                obs = s.Observation(f"obs_{tag}_{prop}", namespace=self.abox)
                obs.observedProperty.append(t[prop])
                obs.hasFeatureOfInterest.append(foi)
                obs.hasSimpleResult.append(float(val))
                obs.resultTime.append(sc.timestamp)
        self._plants[pid] = sc
        self.reasoned = False

    def reason(self) -> float:
        t0 = time.time()
        sync_reasoner_pellet(
            self.world,
            infer_property_values=True,
            infer_data_property_values=True,
            debug=0,
        )
        self.seconds = time.time() - t0
        self.reasoned = True
        return self.seconds

    # -- membaca hasil
    def _definition(self, individual) -> str:
        d = getattr(individual, "definition", None)
        return str(d[0]) if d else ""

    def diagnose(self, plant_id: int) -> Diagnosis:
        if not self.reasoned:
            self.reason()
        t, sc = self.tbox, self._plants[plant_id]
        plant = self.abox[f"plant_{plant_id}"]
        fired_names = {c.name for c in plant.hasCondition}
        full_names = {d.name for d in getattr(plant, "isFullyFavourableFor", [])}
        disease_ind = {code: t[DISEASE_IRI_NAME[code]] for code in DISEASES}
        results, all_agree = {}, True
        for code in DISEASES:
            traces = []
            for r in rules_for(code):
                observed = self._observed_value(sc, r)
                py = r.satisfied_by(observed)
                fired = r.condition in fired_names
                agree = py == fired
                all_agree &= agree
                ind = t[r.condition]
                traces.append(
                    ConditionTrace(
                        r.rule_id, r.condition, code, fired, observed, r.condition_text(),
                        r.swrl(), self._definition(ind), r.source, agree,
                    )
                )
            n_sat = sum(tr.fired for tr in traces)
            sym = disease_ind[code].hasSymptoms
            results[code] = DiseaseResult(
                code=code,
                satisfied=n_sat,
                total=len(traces),
                probability=n_sat / len(traces),
                symptoms=self._definition(sym[0]) if sym else "",
                explanations=[tr.definition for tr in traces if tr.fired],
                traces=traces,
                fully_favourable=DISEASE_IRI_NAME[code] in full_names,
            )
        total_conditions = len(RULES)
        healthy = 1 - sum(r.satisfied for r in results.values()) / total_conditions
        types = sorted(c.name for c in plant.is_a if hasattr(c, "name") and c.name not in ("Plant",))
        return Diagnosis(plant_id, results, healthy, types, self.seconds, all_agree)

    @staticmethod
    def _observed_value(sc: Scenario, r: Rule):
        if r.observed_property is None:
            return bool(sc.rainy_season)
        v = sc.values.get(r.observed_property)
        return None if v is None else float(v)

    def derived_triples(self, plant_id: int) -> list[tuple[str, str, str]]:
        """Fakta baru hasil inferensi untuk satu tanaman (untuk ditampilkan)."""
        plant = self.abox[f"plant_{plant_id}"]
        out = [(plant.name, "hasCondition", c.name) for c in plant.hasCondition]
        out += [(plant.name, "isFullyFavourableFor", d.name)
                for d in getattr(plant, "isFullyFavourableFor", [])]
        out += [(plant.name, "rdf:type", c.name) for c in plant.is_a
                if getattr(c, "name", "") == "HighRiskPlant"]
        return sorted(out)


def diagnose_one(values: Mapping[str, float], rainy_season: bool, level2: bool = True) -> Diagnosis:
    """Diagnosis stateless untuk satu tanaman (dipakai aplikasi Streamlit)."""
    kb = CassavaKB(level2=level2)
    kb.add_scenario(Scenario(1, rainy_season, dict(values)))
    kb.reason()
    d = kb.diagnose(1)
    d.derived = kb.derived_triples(1)
    return d
