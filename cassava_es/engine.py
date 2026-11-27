"""Decision engine (Algoritma 1 paper): gabungkan model visi + model semantik.

Alur (Bagian 3.2 paper):
    1. model visi      : citra            -> vektor probabilitas 5 kelas
    2. model semantik  : data sensor      -> vektor probabilitas 5 kelas (+ penjelasan)
    3. decision engine : weighted majority voting, Eq. (1)  (``decision.combine``)
    4. penjelasan      : diambil dari ontologi untuk kelas pemenang
       (di sini ditambah info umum penyakit dari ``disease_info``)

Mode:
    * fusion         : citra/vektor visi + data sensor  (bentuk utama paper)
    * semantic_only  : hanya data sensor
    * vision_only    : hanya citra/vektor visi (tanpa penjelasan dari ontologi)

Hasil mengikuti bentuk JSON pada Gambar 6 paper (``disease``, ``explanations``,
``visual_certainty``, ``knowledge_certainty``), ditambah info penyakit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Optional

from .decision import combine
from .disease_info import GENERAL_DISCLAIMER, DiseaseInfo, get_info
from .rules import CLASSES, DISEASE_LABEL

SemanticFn = Callable[..., object]  # (values, rainy, level2=...) -> Diagnosis


@dataclass
class FinalResult:
    label: str  # CBB | CBSD | CGM | CMD | Healthy
    probabilities: dict  # vektor akhir per kelas (setelah Eq. 1)
    mode: str  # fusion | semantic_only | vision_only
    weights: dict  # {"semantic": w, "vision": w}
    semantic: Optional[dict]
    vision: Optional[dict]
    visual_certainty: Optional[float]  # p_visi(label)
    knowledge_certainty: Optional[float]  # p_semantik(label)
    visual_supports: bool  # p_visi(label) >= 0.5 ("tanaman menunjukkan gejala visual")
    explanations: list  # definisi kondisi terpenuhi (T-Box) untuk label
    symptoms: str  # teks gejala dari T-Box (kosong bila tidak ada semantik)
    info: DiseaseInfo
    ties: list = field(default_factory=list)
    diagnosis: object = None  # objek Diagnosis (jejak aturan), None bila tanpa semantik

    @property
    def name(self) -> str:
        return DISEASE_LABEL[self.label]

    def to_dict(self) -> dict:
        """Bentuk JSON-able (mirip Gambar 6 paper + info penyakit)."""
        return {
            "disease": self.label,
            "name": self.name,
            "mode": self.mode,
            "probabilities": self.probabilities,
            "weights": self.weights,
            "visual_certainty": self.visual_certainty,
            "knowledge_certainty": self.knowledge_certainty,
            "explanations": list(self.explanations),
            "symptoms_ontology": self.symptoms,
            "ties": list(self.ties),
            "disease_info": asdict(self.info),
            "disclaimer": GENERAL_DISCLAIMER,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)


class DecisionEngine:
    """Decision engine. ``vision`` boleh None (lalu kirim ``vision_vector`` manual)."""

    def __init__(
        self,
        vision=None,
        level2: bool = True,
        semantic_fn: Optional[SemanticFn] = None,
    ):
        self.vision = vision  # objek VisionModel atau None
        self.level2 = level2
        if semantic_fn is None:
            from .kb import diagnose_one  # lazy: butuh owlready2 + Java saat dipanggil

            semantic_fn = diagnose_one
        self._semantic_fn = semantic_fn

    # ------------------------------------------------------------------ API
    def predict(
        self,
        *,
        image=None,
        sensors: Optional[Mapping[str, float]] = None,
        rainy: bool = False,
        vision_vector: Optional[Mapping[str, float]] = None,
        diagnosis=None,
        w_semantic: float = 0.5,
        w_vision: float = 0.5,
    ) -> FinalResult:
        """Hasilkan keputusan akhir.

        image          : citra daun (path/bytes/PIL/file-like) -> diproses model visi.
        vision_vector  : alternatif: vektor probabilitas visi yang sudah ada.
        sensors, rainy : data sensor untuk model semantik (Pellet).
        diagnosis      : hasil ``diagnose_one`` yang sudah dihitung (hindari reasoning ulang).
        """
        vec_vision = self._vision_vector(image, vision_vector)
        if diagnosis is None and sensors is not None:
            diagnosis = self._semantic_fn(dict(sensors), bool(rainy), level2=self.level2)
        if vec_vision is None and diagnosis is None:
            raise ValueError("butuh minimal salah satu: citra/vektor visi atau data sensor")

        semantic = diagnosis.semantic_vector() if diagnosis is not None else None
        if semantic is not None and vec_vision is not None:
            mode, ws, wv = "fusion", w_semantic, w_vision
            final, label = combine(semantic, vec_vision, ws, wv)
        elif semantic is not None:
            mode, ws, wv = "semantic_only", 1.0, 0.0
            final, label = combine(semantic, None)
        else:
            mode, ws, wv = "vision_only", 0.0, 1.0
            final, label = combine(dict.fromkeys(CLASSES, 0.0), vec_vision, 0.0, 1.0)

        ties = [c for c in CLASSES if c != label and abs(final[c] - final[label]) < 1e-12]
        explanations, symptoms = [], ""
        if diagnosis is not None and label != "Healthy":
            res = diagnosis.diseases[label]
            explanations, symptoms = list(res.explanations), res.symptoms

        return FinalResult(
            label=label,
            probabilities=final,
            mode=mode,
            weights={"semantic": ws, "vision": wv},
            semantic=semantic,
            vision=vec_vision,
            visual_certainty=None if vec_vision is None else float(vec_vision[label]),
            knowledge_certainty=None if semantic is None else float(semantic[label]),
            visual_supports=bool(vec_vision is not None and label != "Healthy"
                                 and vec_vision[label] >= 0.5),
            explanations=explanations,
            symptoms=symptoms,
            info=get_info(label),
            ties=ties,
            diagnosis=diagnosis,
        )

    # ------------------------------------------------------------ internal
    def _vision_vector(self, image, vision_vector) -> Optional[dict]:
        if vision_vector is not None:
            missing = [c for c in CLASSES if c not in vision_vector]
            if missing:
                raise KeyError(f"vektor visi kekurangan kelas: {missing}")
            return {c: float(vision_vector[c]) for c in CLASSES}
        if image is None:
            return None
        if self.vision is None:
            raise RuntimeError(
                "citra diberikan tetapi model visi belum dimuat "
                "(latih dengan scripts/train_vision.py lalu muat VisionModel.load)."
            )
        return self.vision.predict_proba(image)
