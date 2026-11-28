"""Model visi (klasifikasi citra daun singkong) untuk decision engine.

Komponen "vision model" pada paper (Bagian 3.2.1): model citra memberi vektor
probabilitas 5 kelas (CBB, CBSD, CGM, CMD, Healthy). Paper memakai VOLO-D2
384x384 dari ``timm``; di sini arsitektur apa pun dari ``timm`` dapat dipakai
dan dicatat di dalam berkas checkpoint.

torch/timm di-import secara LAZY: aplikasi tetap dapat berjalan tanpa torch
(mode semantik saja). Gunakan ``torch_available()`` untuk memeriksanya.

Format checkpoint (``scripts/train_vision.py`` menulisnya):
    {"model_name", "img_size", "mean", "std", "classes", "state_dict", "metrics", "meta"}
Hanya tensor + tipe dasar Python, sehingga dimuat dengan ``weights_only=True``
(tidak mengeksekusi kode dari berkas checkpoint).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageOps

from .rules import CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageLike = Union[str, Path, bytes, Image.Image, Any]  # Any = objek file-like (mis. UploadedFile)


def torch_available() -> bool:
    """True bila torch dan timm dapat di-import."""
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


# ------------------------------------------------------------------ citra
def to_pil(image: ImageLike) -> Image.Image:
    """Terima path, bytes, PIL.Image, atau objek file-like -> PIL RGB.

    Orientasi EXIF diterapkan (foto dari ponsel sering tersimpan miring), lalu
    diubah ke RGB (PNG berkanal alfa / grayscale tidak merusak pra-pemrosesan).
    """
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif hasattr(image, "read"):
        data = image.read()
        if hasattr(image, "seek"):
            try:
                image.seek(0)
            except Exception:  # pragma: no cover
                pass
        img = Image.open(io.BytesIO(data))
    else:
        raise TypeError(f"tipe citra tidak didukung: {type(image)!r}")
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess(image: ImageLike, img_size: int, mean: Sequence[float], std: Sequence[float]):
    """Citra -> tensor float32 [3, H, W] ternormalisasi.

    SATU fungsi ini dipakai oleh inferensi DAN evaluasi di skrip pelatihan,
    supaya pra-pemrosesan uji dan inferensi tidak pernah berbeda.
    """
    import torch

    img = to_pil(image).resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


# ------------------------------------------------------------------ checkpoint
def save_checkpoint(
    path: str | Path,
    model,
    *,
    model_name: str,
    img_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    metrics: Optional[dict] = None,
    meta: Optional[dict] = None,
) -> None:
    import torch

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_name": model_name,
            "img_size": int(img_size),
            "mean": [float(x) for x in mean],
            "std": [float(x) for x in std],
            "classes": list(CLASSES),
            "state_dict": state,
            "metrics": metrics or {},
            "meta": meta or {},
        },
        str(path),
    )


class VisionModel:
    """Pembungkus inferensi: citra -> {kelas: probabilitas}."""

    def __init__(self, model, model_name: str, img_size: int, mean, std, device, meta=None):
        self.model = model
        self.model_name = model_name
        self.img_size = int(img_size)
        self.mean, self.std = list(mean), list(std)
        self.device = device
        self.meta = meta or {}

    # -- memuat
    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "VisionModel":
        if not torch_available():
            raise RuntimeError(
                "torch/timm belum terpasang. Jalankan: pip install -r requirements-vision.txt"
            )
        import timm
        import torch

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint tidak ditemukan: {path}")
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        for key in ("model_name", "img_size", "classes", "state_dict"):
            if key not in ckpt:
                raise KeyError(f"checkpoint tidak valid, kunci '{key}' tidak ada: {path}")
        if list(ckpt["classes"]) != list(CLASSES):
            raise ValueError(
                f"urutan kelas checkpoint {ckpt['classes']} != urutan sistem {CLASSES}; "
                "vektor probabilitas akan tertukar."
            )
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(CLASSES))
        model.load_state_dict(ckpt["state_dict"])
        model.to(dev).eval()
        meta = dict(ckpt.get("meta") or {})
        meta["metrics"] = ckpt.get("metrics") or {}
        return cls(
            model, ckpt["model_name"], ckpt["img_size"],
            ckpt.get("mean", IMAGENET_MEAN), ckpt.get("std", IMAGENET_STD), dev, meta,
        )

    # -- inferensi
    def predict_proba_batch(self, images: Sequence[ImageLike]) -> list[dict]:
        import torch

        batch = torch.stack([preprocess(im, self.img_size, self.mean, self.std) for im in images])
        with torch.inference_mode():
            logits = self.model(batch.to(self.device))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return [{c: float(p[i]) for i, c in enumerate(CLASSES)} for p in probs]

    def predict_proba(self, image: ImageLike) -> dict:
        """Vektor probabilitas {CBB, CBSD, CGM, CMD, Healthy} untuk satu citra."""
        return self.predict_proba_batch([image])[0]

    def describe(self) -> str:
        acc = (self.meta.get("metrics") or {}).get("accuracy")
        extra = f", akurasi uji {acc:.4f}" if isinstance(acc, (int, float)) else ""
        return f"{self.model_name} @ {self.img_size}x{self.img_size} ({self.device}){extra}"
