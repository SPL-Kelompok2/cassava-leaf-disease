"""Basis aturan pakar (IF-THEN) untuk penyakit singkong.

Sumber nilai ambang
-------------------
Seluruh ambang di bawah ini adalah ambang "Lower"/"Upper" pada Tabel 1 paper
Chhetri et al. (2023), Expert Systems with Applications 233, 120955, kecuali
aturan R01 (musim hujan, bertipe boolean) yang tidak ada di Tabel 1 tetapi ada
sebagai kondisi ``CBBRainySeason`` di ontologi paper.

Paper (Bagian 4.1) menyatakan data simulasi dibuat dari "informasi yang
diperoleh dari pakar" dan menyarankan pembaca melihat Luampon &
Charmongkolpradit (2019) dan Seena Radhakrishnan et al. (2022) untuk implikasi
kelembapan/pH/suhu. Paper TIDAK memetakan referensi per aturan, sehingga kolom
``source`` hanya menyebut apa yang benar-benar tertulis di paper.

Setiap aturan mempunyai bentuk SWRL yang dihasilkan dari tabel ini, sehingga
tabel ini adalah satu-satunya sumber kebenaran (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PAPER = "Chhetri et al. (2023), Expert Systems with Applications 233:120955"

# Level fitur yang diamati: sensor tanah menempel pada Soil, sensor udara
# (kelembapan relatif, suhu udara) dan musim hujan menempel pada Field.
SOIL = "soil"
FIELD = "field"


@dataclass(frozen=True)
class Rule:
    rule_id: str  # R01..R14
    disease: str  # CBB | CBSD | CGM | CMD
    condition: str  # nama individu DiseaseCondition di T-Box
    observed_property: Optional[str]  # nama individu sosa:ObservableProperty; None = boolean
    level: str  # SOIL | FIELD
    lower: Optional[float]  # batas bawah (inklusif) atau None
    upper: Optional[float]  # batas atas (inklusif) atau None
    unit: str
    source: str

    # ------------------------------------------------------------------ teks
    def condition_text(self) -> str:
        """Bagian IF dalam bahasa manusia."""
        if self.observed_property is None:
            return "musim hujan = ya"
        name = {
            "SoilMoisture": "kelembapan tanah",
            "SoilPH": "pH tanah",
            "SoilTemperature": "suhu tanah",
            "RelativeHumidity": "kelembapan relatif udara",
            "Temperature": "suhu udara",
        }[self.observed_property]
        if self.lower is not None and self.upper is not None:
            rng = f"{_fmt(self.lower)} <= {name} <= {_fmt(self.upper)}"
        elif self.lower is not None:
            rng = f"{name} >= {_fmt(self.lower)}"
        else:
            rng = f"{name} <= {_fmt(self.upper)}"
        return rng + (f" {self.unit}" if self.unit else "")

    def satisfied_by(self, value) -> bool:
        """Evaluasi murni-Python (dipakai untuk verifikasi silang dan jejak)."""
        if self.observed_property is None:
            return bool(value) is True
        if value is None:
            return False
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True

    # ------------------------------------------------------------------ SWRL
    def swrl(self) -> str:
        """Bentuk SWRL (sintaks owlready2) untuk reasoner Pellet."""
        if self.observed_property is None:
            return (
                "Plant(?plant), isPlantedIn(?plant, ?soil), "
                "isPartOfField(?soil, ?field), hasRainySeason(?field, ?rainy), "
                f"equal(?rainy, true) -> hasCondition(?plant, {self.condition})"
            )
        parts = ["Plant(?plant)", "isPlantedIn(?plant, ?soil)"]
        if self.level == SOIL:
            parts.append("isFeatureOfInterestOf(?soil, ?obs)")
        else:
            parts.append("isPartOfField(?soil, ?field)")
            parts.append("isFeatureOfInterestOf(?field, ?obs)")
        parts.append(f"observedProperty(?obs, {self.observed_property})")
        parts.append("hasSimpleResult(?obs, ?result)")
        if self.lower is not None:
            parts.append(f"greaterThanOrEqual(?result, {_fmt(self.lower)})")
        if self.upper is not None:
            parts.append(f"lessThanOrEqual(?result, {_fmt(self.upper)})")
        return ", ".join(parts) + f" -> hasCondition(?plant, {self.condition})"


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


_T1 = f"Tabel 1 paper ({PAPER}), kolom Lower/Upper"

RULES: list[Rule] = [
    # ---- CBB (Cassava Bacterial Blight)
    Rule("R01", "CBB", "CBBRainySeason", None, FIELD, None, None, "",
         "Kondisi CBBRainySeason pada ontologi paper; bukan baris Tabel 1"),
    Rule("R02", "CBB", "CBBSoilMoisture", "SoilMoisture", SOIL, 0.3, None, "(0-1)", _T1),
    Rule("R03", "CBB", "CBBSoilPH", "SoilPH", SOIL, 6.5, 7.2, "", _T1),
    Rule("R04", "CBB", "CBBSoilTemperature", "SoilTemperature", SOIL, 25, 30, "C", _T1),
    # ---- CBSD (Cassava Brown Streak Disease)
    Rule("R05", "CBSD", "CBSDRelativeHumidity", "RelativeHumidity", FIELD, 0.7, 0.85, "(0-1)", _T1),
    Rule("R06", "CBSD", "CBSDSoilMoisture", "SoilMoisture", SOIL, 0.1, None, "(0-1)", _T1),
    Rule("R07", "CBSD", "CBSDSoilTemperature", "SoilTemperature", SOIL, 10, 32, "C", _T1),
    # ---- CGM (Cassava Green Mottle/Mite; lihat catatan di README)
    Rule("R08", "CGM", "CGMSoilMoisture", "SoilMoisture", SOIL, 0.7, None, "(0-1)", _T1),
    Rule("R09", "CGM", "CGMRelativeHumidity", "RelativeHumidity", FIELD, 0.7, None, "(0-1)", _T1),
    Rule("R10", "CGM", "CGMSoilTemperature", "SoilTemperature", SOIL, 27, None, "C", _T1),
    # ---- CMD (Cassava Mosaic Disease)
    Rule("R11", "CMD", "CMDTemperature", "Temperature", FIELD, 30, None, "C", _T1),
    Rule("R12", "CMD", "CMDSoilMoisture", "SoilMoisture", SOIL, 0.3, None, "(0-1)", _T1),
    Rule("R13", "CMD", "CMDSoilTemperature", "SoilTemperature", SOIL, 20, 32, "C", _T1),
    Rule("R14", "CMD", "CMDHighRainfall", "RelativeHumidity", FIELD, 0.8, None, "(0-1)",
         _T1 + " (baris Relative humidity CMD). Nama kondisi di ontologi = 'CMDHighRainfall' "
         "walau variabel yang diuji kelembapan relatif"),
]

RULES_BY_ID = {r.rule_id: r for r in RULES}
RULES_BY_CONDITION = {r.condition: r for r in RULES}

# Nama individu penyakit di T-Box <-> singkatan
DISEASE_IRI_NAME = {
    "CBB": "CassavaBacterialBlight",
    "CBSD": "CassavaBrownStreakDisease",
    "CGM": "CassavaGreenMottle",
    "CMD": "CassavaMosaicDisease",
}
DISEASES = list(DISEASE_IRI_NAME)  # urutan konsisten dengan Eq. (1) paper
CLASSES = DISEASES + ["Healthy"]

DISEASE_LABEL = {
    "CBB": "Cassava Bacterial Blight (CBB)",
    "CBSD": "Cassava Brown Streak Disease (CBSD)",
    "CGM": "Cassava Green Mottle (CGM)",
    "CMD": "Cassava Mosaic Disease (CMD)",
    "Healthy": "Sehat (Healthy)",
}


def rules_for(disease: str) -> list[Rule]:
    return [r for r in RULES if r.disease == disease]
