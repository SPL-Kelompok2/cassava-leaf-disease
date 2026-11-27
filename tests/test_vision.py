import io

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from cassava_es.rules import CLASSES  # noqa: E402
from cassava_es.vision import (  # noqa: E402
    VisionModel, preprocess, save_checkpoint, to_pil, torch_available,
)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    p = tmp_path_factory.mktemp("ck") / "vision.pt"
    save_checkpoint(p, model, model_name="resnet18", img_size=64,
                    metrics={"accuracy": 0.5, "epoch": 3}, meta={"note": "uji", "n": [1, 2]})
    return p


def _img(w=80, h=60, seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(h, w, 3) * 255).astype("uint8"))


def test_torch_available():
    assert torch_available() is True


def test_muat_checkpoint_dan_prediksi_valid(ckpt):
    vm = VisionModel.load(ckpt, device="cpu")
    assert (vm.model_name, vm.img_size) == ("resnet18", 64)
    p = vm.predict_proba(_img())
    assert list(p) == list(CLASSES)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert "resnet18 @ 64x64" in vm.describe() and "0.5000" in vm.describe()


def test_prediksi_deterministik_dan_batch_sama_dengan_tunggal(ckpt):
    vm = VisionModel.load(ckpt, device="cpu")
    a, b = _img(seed=1), _img(seed=2)
    single = [vm.predict_proba(a), vm.predict_proba(b)]
    batch = vm.predict_proba_batch([a, b])
    for s, t in zip(single, batch):
        for c in CLASSES:
            assert s[c] == pytest.approx(t[c], abs=1e-5)
    assert vm.predict_proba(a) == vm.predict_proba(a)


def test_semua_jenis_masukan_citra_memberi_hasil_sama(ckpt, tmp_path):
    vm = VisionModel.load(ckpt, device="cpu")
    img = _img(seed=3)
    path = tmp_path / "x.png"
    img.save(path)
    raw = path.read_bytes()
    ref = vm.predict_proba(img)
    for src in (path, str(path), raw, io.BytesIO(raw)):
        got = vm.predict_proba(src)
        for c in CLASSES:
            assert got[c] == pytest.approx(ref[c], abs=1e-6)


def test_file_like_dibaca_ulang_setelah_dipakai(ckpt):
    buf = io.BytesIO()
    _img().save(buf, format="PNG")
    buf.seek(0)
    to_pil(buf)
    assert buf.read(4) == b"\x89PNG"  # posisi dikembalikan ke awal


def test_exif_orientasi_diterapkan():
    img = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # putar 90 derajat
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    out = to_pil(buf.getvalue())
    assert out.size == (20, 40)


def test_mode_rgba_dan_grayscale_diubah_ke_rgb():
    for mode in ("RGBA", "L", "P"):
        assert to_pil(Image.new(mode, (10, 10))).mode == "RGB"


def test_preprocess_bentuk_dan_normalisasi():
    t = preprocess(Image.new("RGB", (30, 50), (255, 255, 255)), 32, MEAN, STD)
    assert tuple(t.shape) == (3, 32, 32) and t.dtype == torch.float32
    assert t[0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-5)


def test_tipe_citra_tidak_didukung():
    with pytest.raises(TypeError):
        to_pil(123)


def test_checkpoint_tidak_ada(tmp_path):
    with pytest.raises(FileNotFoundError):
        VisionModel.load(tmp_path / "tidak-ada.pt")


def test_urutan_kelas_berbeda_ditolak(tmp_path):
    p = tmp_path / "salah.pt"
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    torch.save({"model_name": "resnet18", "img_size": 32, "classes": list(reversed(CLASSES)),
                "state_dict": model.state_dict()}, p)
    with pytest.raises(ValueError, match="urutan kelas"):
        VisionModel.load(p, device="cpu")


def test_checkpoint_tanpa_kunci_wajib_ditolak(tmp_path):
    p = tmp_path / "rusak.pt"
    torch.save({"model_name": "resnet18"}, p)
    with pytest.raises(KeyError):
        VisionModel.load(p, device="cpu")


def test_checkpoint_berbahaya_tidak_dieksekusi(tmp_path):
    """weights_only=True: objek Python sembarang (pickle) harus ditolak, bukan dijalankan."""
    p = tmp_path / "jahat.pt"

    class Evil:
        def __reduce__(self):
            return (__import__("os").system, ("echo PWNED > " + str(tmp_path / "pwned.txt"),))

    torch.save({"model_name": "resnet18", "img_size": 32, "classes": list(CLASSES),
                "state_dict": {}, "meta": Evil()}, p)
    with pytest.raises(Exception):
        VisionModel.load(p, device="cpu")
    assert not (tmp_path / "pwned.txt").exists()
