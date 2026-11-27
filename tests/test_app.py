import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from conftest import needs_java

pytestmark = needs_java
APP = str(Path(__file__).resolve().parent.parent / "app.py")
GO = "🔍 Diagnosa sekarang"
VS_NONE, VS_PHOTO, VS_MANUAL = "Tanpa foto", "📷 Foto daun", "Vektor manual (pengujian)"


def _new():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


def _go(at):
    [b for b in at.button if b.label == GO][0].click().run()
    assert not at.exception
    return at


def _run(preset):
    at = _new()
    at.radio(key="vision_source").set_value(VS_NONE).run()
    at.selectbox(key="preset").select(preset).run()
    return _go(at)


def _subheaders(at):
    return [s.value for s in at.subheader]


def _all_markdown(at):
    return "\n".join(m.value for m in at.markdown)


# ---------------------------------------------------------------- perilaku lama (semantik saja)
def test_app_awal_dan_preset_cmd():
    at = _new()
    assert any(GO in i.value for i in at.info)
    at = _run("Contoh CMD")
    assert any("Cassava Mosaic Disease (CMD)" in s for s in _subheaders(at))
    assert not at.error


def test_app_preset_sehat():
    at = _run("Contoh tanpa kondisi pendukung")
    assert any("Sehat" in s for s in _subheaders(at))


def test_app_menampilkan_info_umum_penyakit():
    at = _run("Contoh CMD")
    md = _all_markdown(at)
    assert "Penyakit mosaik singkong" in md and "Pengendalian umum" in md and "Penyebab:" in md
    assert "Tentang penyakit ini" in md
    assert any("bukan diagnosis" in c.value for c in at.caption)


def test_app_info_cgm_menjelaskan_tungau():
    at = _run("Contoh CGM")
    md = _all_markdown(at)
    assert "Tungau hijau singkong" in md and "Mononychellus tanajoa" in md


def test_app_info_sehat_punya_peringatan_keterbatasan():
    at = _run("Contoh tanpa kondisi pendukung")
    md = _all_markdown(at)
    warn = "\n".join(w.value for w in at.warning)
    assert "Tentang kondisi ini" in md and "bukan jaminan" in warn


# ---------------------------------------------------------------- vektor visi manual (Eq. 1)
def test_app_dengan_vektor_visi_manual_mengubah_keputusan():
    at = _new()
    at.selectbox(key="preset").select("Contoh CMD").run()
    at.radio(key="vision_source").set_value(VS_MANUAL).run()
    for c, v in {"CBB": 0.0, "CBSD": 0.0, "CGM": 0.0, "CMD": 0.0, "Healthy": 1.0}.items():
        at.number_input(key=f"v-{c}").set_value(v)
    _go(at)
    # semantik CMD=1.0, Healthy=1-9/14=0.357; visi Healthy=1.0; bobot sama -> Healthy 0.679 > CMD 0.5
    assert any("Sehat" in s for s in _subheaders(at))
    assert not any("Cassava Mosaic" in s for s in _subheaders(at) if s.startswith("Kesimpulan"))
    assert at.metric[0].value != "-" and at.metric[1].value != "-"


def test_app_bobot_mengubah_keputusan_tanpa_menjalankan_ulang_pellet():
    at = _new()
    at.selectbox(key="preset").select("Contoh CMD").run()
    at.radio(key="vision_source").set_value(VS_MANUAL).run()
    for c, v in {"CBB": 0.0, "CBSD": 0.0, "CGM": 0.0, "CMD": 0.0, "Healthy": 1.0}.items():
        at.number_input(key=f"v-{c}").set_value(v)
    _go(at)
    assert any("Sehat" in s for s in _subheaders(at))
    at.number_input(key="w_sem").set_value(5.0).run()  # semantik jauh lebih berbobot
    assert any("Cassava Mosaic Disease (CMD)" in s for s in _subheaders(at))


def test_app_visi_saja_tanpa_sensor():
    at = _new()
    at.checkbox(key="use_sensors").uncheck().run()
    at.radio(key="vision_source").set_value(VS_MANUAL).run()
    for c, v in {"CBB": 0.0, "CBSD": 0.9, "CGM": 0.0, "CMD": 0.1, "Healthy": 0.0}.items():
        at.number_input(key=f"v-{c}").set_value(v)
    _go(at)
    assert any("Cassava Brown Streak Disease (CBSD)" in s for s in _subheaders(at))
    assert "tanpa penjelasan dari ontologi" in _all_markdown(at) or any(
        "tanpa penjelasan dari ontologi" in m.value for m in at.markdown)


def test_app_tanpa_sumber_apa_pun_menampilkan_error():
    at = _new()
    at.checkbox(key="use_sensors").uncheck().run()
    at.radio(key="vision_source").set_value(VS_NONE).run()
    _go(at)
    assert any("minimal satu sumber" in e.value for e in at.error)


# ---------------------------------------------------------------- foto + model terlatih
def _tiny_checkpoint(path):
    timm = pytest.importorskip("timm")
    from cassava_es.vision import save_checkpoint

    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    save_checkpoint(path, model, model_name="resnet18", img_size=32, metrics={"accuracy": 0.5})


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.fromarray((np.random.RandomState(0).rand(40, 30, 3) * 255).astype("uint8")).save(buf, format="JPEG")
    return buf.getvalue()


def test_app_foto_tanpa_unggah_menampilkan_error(tmp_path):
    ck = tmp_path / "v.pt"
    _tiny_checkpoint(ck)
    at = _new()
    at.radio(key="vision_source").set_value(VS_PHOTO).run()
    at.text_input(key="ckpt_path").set_value(str(ck)).run()
    _go(at)
    assert any("Unggah foto" in e.value for e in at.error)


def test_app_foto_checkpoint_tidak_ada_memberi_peringatan(tmp_path):
    at = _new()
    at.radio(key="vision_source").set_value(VS_PHOTO).run()
    at.text_input(key="ckpt_path").set_value(str(tmp_path / "tidak-ada.pt")).run()
    assert any("Checkpoint tidak ditemukan" in w.value for w in at.sidebar.warning)


def test_app_foto_penuh_fusi_dengan_sensor(tmp_path):
    ck = tmp_path / "v.pt"
    _tiny_checkpoint(ck)
    at = _new()
    at.selectbox(key="preset").select("Contoh CMD").run()
    at.radio(key="vision_source").set_value(VS_PHOTO).run()
    at.text_input(key="ckpt_path").set_value(str(ck)).run()
    at.file_uploader(key="photo").upload("daun.jpg", _jpeg_bytes(), "image/jpeg").run()
    _go(at)
    assert not at.error
    assert any(s.startswith("Kesimpulan:") for s in _subheaders(at))
    assert at.metric[0].value != "-" and at.metric[1].value != "-"  # keyakinan visual & pengetahuan
    assert any("Weighted majority voting" in m.value for m in at.markdown)
