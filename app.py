"""Aplikasi Streamlit: sistem pakar penyakit singkong (ontologi + SWRL + Pellet)
digabung dengan model visi (foto daun) lewat decision engine (Eq. 1 paper).

Jalankan:  streamlit run app.py
Membutuhkan Java (untuk reasoner Pellet milik owlready2). Model visi bersifat
opsional dan butuh torch/timm + checkpoint dari scripts/train_vision.py.

Tata letak UI (terinspirasi aplikasi Plantix/Agrio):
    Diagnosa         : foto daun -> kondisi lahan (bertumpuk) -> hasil di jendela pop-up
    Pustaka penyakit : kartu info tiap kelas
    Mode pakar       : jejak aturan SWRL, fakta turunan Pellet, knowledge graph T-Box/A-Box,
                       basis aturan & CQ
    Tentang          : metode, kinerja model, keterbatasan
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from cassava_es import cq, kgraph
from cassava_es.disease_info import DISEASE_INFO, GENERAL_DISCLAIMER
from cassava_es.engine import DecisionEngine
from cassava_es.kb import diagnose_one
from cassava_es.presets import PRESETS
from cassava_es.rules import CLASSES, DISEASE_LABEL, RULES
from cassava_es.vision import VisionModel, torch_available

st.set_page_config(page_title="Sistem Pakar Penyakit Singkong", page_icon="🌿", layout="centered")

ROOT = Path(__file__).resolve().parent
PROPS = ("SoilMoisture", "SoilPH", "SoilTemperature", "RelativeHumidity", "Temperature")
LABELS = {
    "SoilMoisture": "💧 Kelembapan tanah (0-1)",
    "SoilPH": "🧪 pH tanah (0-14)",
    "SoilTemperature": "🌡️ Suhu tanah (°C)",
    "RelativeHumidity": "☁️ Kelembapan udara (0-1)",
    "Temperature": "☀️ Suhu udara (°C)",
}
HELP = {
    "SoilMoisture": "Fraksi kadar air tanah: 0 = sangat kering, 1 = jenuh air.",
    "SoilPH": "Derajat keasaman tanah. 7 = netral.",
    "SoilTemperature": "Suhu tanah di zona perakaran.",
    "RelativeHumidity": "Kelembapan relatif udara dalam fraksi: 0,8 berarti 80%.",
    "Temperature": "Suhu udara di sekitar tanaman.",
}
BOUNDS = {  # batas isian UI = rentang Healthy/Min-Max Tabel 1 diperluas sedikit
    "SoilMoisture": (0.0, 1.0, 0.01), "SoilPH": (0.0, 14.0, 0.1),
    "SoilTemperature": (0.0, 60.0, 0.5), "RelativeHumidity": (0.0, 1.0, 0.01),
    "Temperature": (0.0, 60.0, 0.5),
}

# Warna & ikon per kelas (dipakai konsisten di seluruh UI)
STYLE = {
    "CBB": {"color": "#C0392B", "icon": "🦠"},
    "CBSD": {"color": "#8D5524", "icon": "🍂"},
    "CGM": {"color": "#D68910", "icon": "🕷️"},
    "CMD": {"color": "#7D3C98", "icon": "🧬"},
    "Healthy": {"color": "#1E8E3E", "icon": "🌱"},
}

VS_PHOTO = "Foto daun"
VS_NONE = "Tanpa foto"
VS_MANUAL = "Vektor manual (pengujian)"
GO = "Diagnosa sekarang"
VS_ICON = {"Foto daun": ":material/photo_camera:", "Tanpa foto": ":material/block:",
           "Vektor manual (pengujian)": ":material/tune:"}
DEFAULT_CKPT = os.environ.get("CASSAVA_VISION_CKPT", "checkpoints/vision.pt")
METRICS_FILE = ROOT / "notebooks" / "output" / "vision" / "vision_metrics.json"

CSS = """
<style>
.block-container {max-width: 980px;}
h1, h2, h3, h4, h5 {letter-spacing: -0.02em;}

/* ---------- hero */
.hero {position: relative; overflow: hidden; text-align: center; color: #fff;
       background: radial-gradient(120% 140% at 50% 0%, #136B3E 0%, #0A3D24 55%, #062818 100%);
       border-radius: 28px; padding: 44px 32px 30px; margin-bottom: 18px;
       box-shadow: 0 1px 0 rgba(255,255,255,.08) inset, 0 24px 48px -24px rgba(6,40,24,.55);}
.hero::after {content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .35;
       background-image: radial-gradient(rgba(255,255,255,.18) 1px, transparent 1px);
       background-size: 22px 22px; -webkit-mask-image: linear-gradient(180deg, #000 0%, transparent 85%);
       mask-image: linear-gradient(180deg, #000 0%, transparent 85%);}
.hero .glow {position: absolute; border-radius: 50%; filter: blur(60px); pointer-events: none;}
.hero .g1 {width: 340px; height: 340px; background: #9BE15D; opacity: .28; top: -160px; left: -80px;}
.hero .g2 {width: 300px; height: 300px; background: #2BD4A4; opacity: .22; bottom: -180px; right: -60px;}
.hero > * {position: relative; z-index: 1;}
.eyebrow {display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
          font-size: .78rem; font-weight: 600; letter-spacing: .02em; color: #DDF7C8;
          background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16);
          backdrop-filter: blur(8px);}
.eyebrow i {width: 7px; height: 7px; border-radius: 50%; background: #A3E635;
            box-shadow: 0 0 0 4px rgba(163,230,53,.25);}
.hero h1 {color: #fff; font-size: clamp(1.9rem, 4.6vw, 2.9rem); font-weight: 800; line-height: 1.08;
          margin: 16px 0 10px; padding: 0; letter-spacing: -0.035em;}
.hero h1 em {font-style: normal; background: linear-gradient(90deg, #BEF264, #5EEAD4);
             -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero p {color: rgba(232,247,236,.82); margin: 0 auto; font-size: 1.04rem; max-width: 600px; line-height: 1.55;}
.stats {display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 24px;}
.stats > div {padding: 10px 18px; border-radius: 16px; background: rgba(255,255,255,.06);
              border: 1px solid rgba(255,255,255,.12); min-width: 128px;}
.stats b {display: block; font-size: 1.2rem; color: #fff; font-weight: 800;}
.stats span {font-size: .74rem; color: rgba(232,247,236,.7);}

/* ---------- kartu input */
[class*="st-key-card_"] {background: #fff; border-radius: 22px !important; padding: 22px 24px !important;
       border: 1px solid #E3EADD !important;
       box-shadow: 0 1px 2px rgba(20,33,26,.04), 0 12px 32px -18px rgba(20,33,26,.18);}
.sec {display: flex; align-items: center; gap: 12px; margin: 0 0 6px;}
.sec .n {flex: 0 0 34px; height: 34px; border-radius: 11px; display: grid; place-items: center;
         background: linear-gradient(135deg, #0F7A43, #34A853); color: #fff; font-weight: 800; font-size: .92rem;
         box-shadow: 0 6px 14px -6px rgba(15,122,67,.6);}
.sec .t {font-weight: 750; font-size: 1.08rem; line-height: 1.2;}
.sec .s {color: #6B7A70; font-size: .84rem;}
.st-key-photo section {border: 1.5px dashed #A9C99B !important; background: #F5FAF0 !important;
       border-radius: 18px !important; transition: border-color .2s, background .2s;}
.st-key-photo section:hover {border-color: #0F7A43 !important; background: #EEF7E6 !important;}

/* ---------- tombol utama */
.st-key-go button {background: linear-gradient(135deg, #0F7A43 0%, #22A05A 100%) !important; border: 0 !important;
       min-height: 3rem; font-weight: 700; letter-spacing: .01em;
       box-shadow: 0 10px 24px -10px rgba(15,122,67,.7); transition: transform .15s, box-shadow .15s;}
.st-key-go button:hover {transform: translateY(-1px); box-shadow: 0 14px 30px -10px rgba(15,122,67,.8);}

/* ---------- elemen umum */
.muted {color: #6B7A70; font-size: .88rem;}
.pill {display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .74rem;
       font-weight: 700; margin-right: 6px; border: 1px solid transparent; letter-spacing: .01em;}

/* ---------- hasil */
.res-head {display: flex; align-items: center; gap: 20px; border-radius: 20px; padding: 18px 22px;
           margin: 4px 0 8px; background: linear-gradient(135deg, var(--ct), transparent 80%);
           border: 1px solid var(--cb);}
.res-main {flex: 1; min-width: 0;}
.res-head .big {font-size: 1.6rem; font-weight: 800; line-height: 1.2; margin: 10px 0 2px; color: var(--c);
                letter-spacing: -0.02em;}
.res-head .en {font-style: italic; color: #5A6A5F; font-size: .9rem;}
.lvl {display: inline-flex; align-items: center; gap: 7px; margin-top: 10px; font-size: .82rem; font-weight: 700;
      color: var(--l);}
.lvl i {width: 8px; height: 8px; border-radius: 50%; background: var(--l);}
.ring {flex: 0 0 112px; height: 112px; border-radius: 50%; display: grid; place-items: center;
       background: conic-gradient(var(--l) calc(var(--p) * 1%), #E6ECE1 0);}
.ring > div {width: 88px; height: 88px; border-radius: 50%; background: #fff; display: grid; place-content: center;
             text-align: center;}
.ring b {font-size: 1.55rem; font-weight: 800; line-height: 1; color: #14211A;}
.ring span {font-size: .66rem; color: #6B7A70; margin-top: 3px;}
.rank {display: grid; grid-template-columns: 30px 1fr 54px; align-items: center; gap: 10px;
       margin: 9px 0; font-size: .92rem;}
.rank .track {height: 8px; background: #EAF0E5; border-radius: 999px; overflow: hidden; margin-top: 5px;}
.rank .track > div {height: 100%; border-radius: 999px;}
.rank .v {text-align: right; font-variant-numeric: tabular-nums; font-weight: 700;}
.chk {display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; border-radius: 14px;
      margin: 6px 0; background: #fff; border: 1px solid #E6ECE1; font-size: .93rem;}
.chk.ok {background: #F0F9EE; border-color: #C5E5BD;}
.chk .obs {margin-left: auto; color: #6B7A70; white-space: nowrap; font-variant-numeric: tabular-nums;}
.act {display: flex; gap: 12px; align-items: flex-start; padding: 12px 16px; border-radius: 16px;
      background: #fff; border: 1px solid #E6ECE1; margin: 8px 0;}
.act .n {flex: 0 0 28px; height: 28px; border-radius: 9px; display: grid; place-items: center;
         background: #E7F4E4; color: #0F7A43; font-weight: 800; font-size: .85rem;}

/* ---------- pustaka */
.lib {position: relative; border-radius: 20px; background: #fff; border: 1px solid #E3EADD; padding: 18px 16px;
      height: 100%; overflow: hidden; transition: transform .18s, box-shadow .18s;
      box-shadow: 0 1px 2px rgba(20,33,26,.04);}
.lib::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 4px; background: var(--c);}
.lib:hover {transform: translateY(-3px); box-shadow: 0 16px 32px -18px rgba(20,33,26,.3);}
.lib .ic {width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; font-size: 1.4rem;
          background: var(--ct); margin-bottom: 10px;}
.lib .t {font-weight: 750; margin-top: 8px; line-height: 1.25;}
.lib .d {color: #55655A; font-size: .84rem; margin-top: 8px; line-height: 1.5;}
.foot {color: #8A988E; font-size: .78rem; text-align: center; margin-top: 36px;}
</style>
"""


# ------------------------------------------------------------------ cache
@st.cache_data(show_spinner="Menjalankan reasoner Pellet ...")
def run_inference(values: tuple, rainy: bool, level2: bool):
    return diagnose_one(dict(values), rainy, level2=level2)


@st.cache_resource(show_spinner="Memuat model visi ...")
def load_vision(path: str, mtime: float) -> VisionModel:
    return VisionModel.load(path)


@st.cache_data(show_spinner="Menganalisis foto daun ...")
def predict_image(data: bytes, path: str, mtime: float) -> dict:
    return load_vision(path, mtime).predict_proba(data)


@st.cache_data
def vision_metrics() -> dict | None:
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ helper tampilan
def esc(x) -> str:
    return html.escape(str(x))


def tint(hex_color: str, alpha: str = "1A") -> str:
    """Warna transparan dari warna kelas (alpha hex 2 digit)."""
    return hex_color + alpha


def pill(text: str, color: str) -> str:
    return (f'<span class="pill" style="background:{tint(color)};color:{color};'
            f'border-color:{tint(color, "40")}">{esc(text)}</span>')


def html_block(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def pretty_cond(text: str) -> str:
    return text.replace("<=", "≤").replace(">=", "≥")


def fmt_obs(v) -> str:
    if v is None:
        return "tidak ada data"
    if v is True:
        return "ya"
    if v is False:
        return "tidak"
    return f"{v:g}"


def level_text(p: float) -> tuple[str, str]:
    if p >= 0.7:
        return "Kecocokan tinggi", "#1E8E3E"
    if p >= 0.4:
        return "Kecocokan sedang", "#D68910"
    return "Kecocokan rendah", "#C0392B"


def section(num: str, title: str, sub: str = "") -> None:
    html_block(f'<div class="sec"><span class="n">{num}</span><div><div class="t">{esc(title)}</div>'
               f'<div class="s">{esc(sub)}</div></div></div>')


def render_ranking(final: dict, label: str) -> None:
    rows = []
    for c in sorted(CLASSES, key=lambda c: -final[c]):
        s = STYLE[c]
        w = max(0.0, min(1.0, final[c])) * 100
        weight = "700" if c == label else "500"
        rows.append(
            f'<div class="rank"><div>{s["icon"]}</div>'
            f'<div><div style="font-weight:{weight}">{esc(DISEASE_INFO[c].name_id)} '
            f'<span class="muted">· {esc(c)}</span></div>'
            f'<div class="track"><div style="width:{w:.1f}%;background:{s["color"]}"></div></div></div>'
            f'<div class="v">{final[c] * 100:.0f}%</div></div>')
    html_block("".join(rows))


def render_about(info, label: str) -> None:
    """Bagian 'Tentang': definisi, penyebab, gejala, penyebaran, dampak."""
    st.markdown("#### " + ("Tentang kondisi ini" if label == "Healthy" else "Tentang penyakit ini"))
    st.markdown(f"**{info.name_id}** · *{info.name_en}* · {info.category}")
    st.write(info.definition)
    st.markdown(f"**Penyebab:** {info.agent}")
    c1, c2 = st.columns(2)
    with c1:
        if info.symptoms:
            st.markdown("**🔎 Ciri yang tampak**")
            st.markdown("\n".join(f"- {x}" for x in info.symptoms))
    with c2:
        if info.spread:
            st.markdown("**🔁 Cara menyebar**")
            st.markdown("\n".join(f"- {x}" for x in info.spread))
    if info.favourable != "-":
        st.markdown(f"**🌦️ Kondisi yang mendukung:** {info.favourable}")
    if info.impact != "-":
        st.markdown(f"**📉 Dampak:** {info.impact}")


def render_management(info) -> None:
    st.markdown("#### Pengendalian umum")
    html_block("".join(f'<div class="act"><div class="n">{i}</div><div>{esc(x)}</div></div>'
                       for i, x in enumerate(info.management, 1)))
    for n in info.notes:
        st.warning(n, icon="⚠️")
    st.info("Untuk memastikan penyakit dan menentukan penanganan, hubungi **penyuluh pertanian** atau "
            "**dinas pertanian** setempat.", icon="🧑‍🌾")
    if info.sources:
        st.markdown("**Sumber**")
        st.markdown("\n".join(f"- [{t}]({u})" for t, u in info.sources))


def render_info(info) -> None:
    """Kartu info lengkap (dipakai di Pustaka penyakit)."""
    with st.container(border=True):
        render_about(info, info.code)
        render_management(info)
    st.caption(GENERAL_DISCLAIMER)


# ------------------------------------------------------------------ dialog hasil
@st.dialog("Hasil diagnosa", width="large", icon=":material/medical_information:")
def show_result(out, res, snap, w_sem: float, w_vis: float) -> None:
    """Pop-up hasil: kesimpulan, peringkat, dan detail (tentang/pengendalian/alasan/angka)."""
    label, final, sem, vision = out.label, out.probabilities, out.semantic, out.vision
    info, s = out.info, STYLE[label]
    lvl, lvl_color = level_text(final[label])

    # ---- kartu hasil utama
    with st.container(border=True):
        if snap["photo"] is not None:
            top_l, top_r = st.columns([3, 2])
        else:
            top_l, top_r = st.container(), None
        with top_l:
            st.subheader(f"Kesimpulan: {DISEASE_LABEL[label]}")
            pct = max(0.0, min(1.0, final[label])) * 100
            html_block(
                f'<div class="res-head" style="--c:{s["color"]};--ct:{tint(s["color"], "14")};'
                f'--cb:{tint(s["color"], "33")}">'
                f'<div class="res-main"><div>{pill(info.category, s["color"])}{pill(label, s["color"])}</div>'
                f'<div class="big">{s["icon"]} {esc(info.name_id)}</div>'
                f'<div class="en">{esc(info.name_en)}</div>'
                f'<span class="lvl" style="--l:{lvl_color}"><i></i>{lvl}</span></div>'
                f'<div class="ring" style="--p:{pct:.1f};--l:{lvl_color}"><div><b>{pct:.0f}%</b>'
                f'<span>skor akhir</span></div></div></div>')
        if top_r is not None:
            with top_r:
                st.image(snap["photo"], caption="Foto yang dianalisis", width="stretch")

        m1, m2 = st.columns(2)
        m1.metric(":material/photo_camera: Keyakinan visual", "-" if out.visual_certainty is None else f"{out.visual_certainty:.3f}",
                  help="p_visi(kelas terpilih) dari model visi.")
        m2.metric(":material/psychology: Keyakinan pengetahuan", "-" if out.knowledge_certainty is None
                  else f"{out.knowledge_certainty:.3f}",
                  help="p_semantik(kelas terpilih): rasio kondisi lahan yang terpenuhi.")

        if out.mode == "semantic_only":
            st.write("Kesimpulan = kelas dengan skor semantik tertinggi (foto daun tidak digunakan).")
        elif out.mode == "vision_only":
            st.write("Kesimpulan = kelas dengan probabilitas visi tertinggi (data lahan tidak digunakan; "
                     "tanpa penjelasan dari ontologi).")
        else:
            st.write(f"Weighted majority voting (Eq. 1): bobot semantik={w_sem}, bobot visi={w_vis}.")

        if vision is not None:
            top_v = max(CLASSES, key=lambda c: (vision[c], -CLASSES.index(c)))
            if top_v != label:
                st.info(f"Foto daun sendiri mengarah ke **{DISEASE_LABEL[top_v]}** (p={vision[top_v]:.3f}); "
                        "keputusan akhir berbeda karena digabung dengan data lahan menurut bobot.", icon="🔀")
        if out.ties:
            st.warning("Ada kelas lain dengan skor sama persis (" + ", ".join(out.ties) +
                       "); argmax mengambil kelas pertama menurut urutan CBB, CBSD, CGM, CMD, Healthy.")
        if res is not None and not res.consistent:
            st.error("Pellet dan evaluasi Python tidak sepakat pada sebagian aturan; periksa Mode pakar.")

    # ---- kemungkinan lain
    with st.container(border=True):
        st.markdown("**:material/leaderboard: Peringkat kemungkinan**")
        render_ranking(final, label)

    # ---- detail
    t_about, t_mgmt, t_why, t_num = st.tabs(
        [":material/info: Tentang", ":material/shield: Pengendalian", ":material/lightbulb: Mengapa hasil ini?",
         ":material/calculate: Rincian angka"])
    with t_about:
        render_about(info, label)
    with t_mgmt:
        render_management(info)
    with t_why:
        st.markdown("#### Penjelasan untuk pengguna")
        if res is None:
            st.write("Tidak ada data lahan, jadi tidak ada penjelasan berbasis kondisi lingkungan.")
        elif label == "Healthy":
            st.write("Tidak ada penyakit yang kondisi pendukungnya mendominasi pada data ini.")
        else:
            st.write("Kondisi lahan dicocokkan dengan aturan pakar (SWRL) untuk penyakit ini:")
            html_block("".join(
                f'<div class="chk {"ok" if t.fired else ""}"><div>{"✅" if t.fired else "⬜"}</div>'
                f'<div>{esc(pretty_cond(t.if_text))}</div>'
                f'<div class="obs">teramati: {esc(fmt_obs(t.observed))}</div></div>'
                for t in res.diseases[label].traces))
            if out.explanations:
                st.markdown("**Arti kondisi yang terpenuhi (dari ontologi):**")
                st.markdown("\n".join(f"- {e}" for e in out.explanations))
            else:
                st.write("Tidak ada kondisi lahan yang terpenuhi untuk penyakit ini.")
            st.write(f"**Gejala yang biasanya tampak.** {out.symptoms}")
        if res is not None:
            with st.container(border=True):
                st.markdown("**Kecocokan kondisi lahan per penyakit**")
                for c in ("CBB", "CBSD", "CGM", "CMD"):
                    d = res.diseases[c]
                    st.progress(d.satisfied / d.total if d.total else 0.0,
                                text=f"{STYLE[c]['icon']} {DISEASE_INFO[c].name_id}: "
                                     f"{d.satisfied}/{d.total} kondisi")
        if out.visual_supports:
            st.success("Foto daun menunjukkan gejala visual yang sesuai.", icon="📷")

    with t_num:
        cols = {}
        if sem is not None:
            cols["Semantik"] = sem
        if vision is not None:
            cols["Visi"] = vision
        if sem is not None and vision is not None:
            cols["Gabungan"] = final
        st.bar_chart(pd.DataFrame(cols).loc[list(CLASSES)], height=260)
        rows = []
        for c in CLASSES:
            row = {"Kelas": DISEASE_LABEL[c]}
            if res is not None:
                row["Kondisi terpenuhi"] = (f"{res.diseases[c].satisfied}/{res.diseases[c].total}"
                                            if c != "Healthy" else "-")
            if sem is not None:
                row["p semantik"] = round(sem[c], 3)
            if vision is not None:
                row["p visi"] = round(vision[c], 3)
            row["p akhir"] = round(final[c], 3)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if sem is not None:
            st.caption("p semantik(penyakit) = kondisi terpenuhi / jumlah kondisi penyakit; p(sehat) = 1 − "
                       "(seluruh kondisi terpenuhi / 14). Rumus ini mengikuti kode repositori asli; ini skor "
                       "kesesuaian kondisi, bukan probabilitas terkalibrasi.")
        with st.expander("Keluaran JSON decision engine (bentuk seperti Gambar 6 paper)"):
            st.json(out.to_dict(), expanded=False)

    st.caption(GENERAL_DISCLAIMER)


# ------------------------------------------------------------------ kerangka halaman
html_block(CSS)
html_block(
    '<div class="hero"><div class="glow g1"></div><div class="glow g2"></div>'
    '<span class="eyebrow"><i></i>Vision AI · Ontologi OWL · Penalaran SWRL</span>'
    '<h1>Sistem Pakar <em>Penyakit Singkong</em></h1>'
    '<p>Kenali penyakit daun singkong dari foto dan kondisi lahan, lengkap dengan alasan dan '
    'saran pengendalian.</p>'
    '<div class="stats"><div><b>5</b><span>kelas dikenali</span></div>'
    '<div><b>14</b><span>aturan pakar SWRL</span></div>'
    '<div><b>Eq. 1</b><span>fusi visi + semantik</span></div></div></div>'
)

tab_diag, tab_lib, tab_expert, tab_about = st.tabs(
    [":material/stethoscope: Diagnosa", ":material/local_library: Pustaka penyakit",
     ":material/account_tree: Mode pakar (XAI)", ":material/info: Tentang"])

# ================================================================== DIAGNOSA: input
model_note = None
with tab_diag:
    with st.container(border=True, key="card_photo"):
        section("1", "Foto daun", "Unggah atau potret satu daun yang bergejala.")
        vision_source = st.segmented_control(
            "Sumber prediksi visual", [VS_PHOTO, VS_NONE, VS_MANUAL], default=VS_PHOTO, required=True,
            key="vision_source", label_visibility="collapsed", format_func=lambda v: VS_ICON[v] + " " + v)
        photo, ckpt_path, ckpt_mtime, vision_vec_manual = None, DEFAULT_CKPT, 0.0, None
        if vision_source == VS_PHOTO:
            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                use_cam = st.toggle("Ambil langsung dari kamera", key="use_camera")
                if use_cam:
                    photo = st.camera_input("Arahkan kamera ke satu daun", key="camera")
                else:
                    photo = st.file_uploader("Unggah foto daun singkong (JPG/PNG)",
                                             type=["jpg", "jpeg", "png"], key="photo")
                    if photo is not None:
                        st.image(photo, width=220)
                st.caption(":material/tips_and_updates: Satu daun memenuhi bingkai, fokus tajam, cahaya "
                           "terang, bagian bergejala terlihat jelas.", text_alignment="center")
                model_note = st.empty()
        elif vision_source == VS_MANUAL:
            st.caption("Masukkan vektor probabilitas dari model visi mana pun untuk menguji Eq. (1).")
            mc = st.columns(5)
            vision_vec_manual = {c: mc[i].number_input(f"p_visi({c})", 0.0, 1.0, 0.2, 0.01, key=f"v-{c}")
                                 for i, c in enumerate(CLASSES)}
        else:
            st.caption("Diagnosa hanya memakai kondisi lahan (model semantik).")

    with st.container(border=True, key="card_field"):
        section("2", "Kondisi lahan", "Isi manual atau pakai contoh data untuk mencoba.")
        use_sensors = st.checkbox("Pakai data kondisi lahan (sensor)", value=True, key="use_sensors")
        h1, h2 = st.columns([2, 1], vertical_alignment="bottom")
        preset = h1.selectbox("Isi cepat dengan contoh data", list(PRESETS), key="preset",
                              disabled=not use_sensors)
        p = PRESETS[preset] or {}
        rainy = h2.checkbox("🌧️ Sedang musim hujan di ladang", value=bool(p.get("rainy", False)),
                            disabled=not use_sensors)
        vals = {}
        grid = st.columns(3)
        for i, prop in enumerate(PROPS):
            lo, hi, step = BOUNDS[prop]
            default = float(p.get(prop, (lo + hi) / 2))
            vals[prop] = grid[i % 3].number_input(
                LABELS[prop], lo, hi, default, step, key=f"{preset}-{prop}",
                disabled=not use_sensors, help=HELP[prop])
        st.caption(":material/science: Data pada demo ini adalah masukan/simulasi, bukan pengukuran lapangan.")

    # ---- pengaturan lanjutan (dulu di sidebar)
    with st.expander("Pengaturan lanjutan (penguji & pakar)", icon=":material/tune:"):
        st.caption("Pengguna umum tidak perlu mengubah apa pun di sini.")
        vision_ready = False
        if vision_source == VS_PHOTO:
            st.markdown("**Model visi**")
            ckpt_path = st.text_input("Path checkpoint", DEFAULT_CKPT, key="ckpt_path",
                                      help="Hasil scripts/train_vision.py (checkpoints/vision.pt).")
            if not torch_available():
                st.warning("torch/timm belum terpasang: `pip install -r requirements-vision.txt`.")
            elif not Path(ckpt_path).is_file():
                st.warning(f"Checkpoint tidak ditemukan: `{ckpt_path}`. Latih dulu dengan "
                           "`python scripts/train_vision.py ...` (lihat README).")
            else:
                ckpt_mtime = Path(ckpt_path).stat().st_mtime
                vision_ready = True
                st.success("Model visi siap.", icon=":material/check_circle:")
        w_sem, w_vis = 0.5, 0.5
        if vision_source != VS_NONE and use_sensors:
            st.markdown("**Bobot penggabungan (Eq. 1)** · Paper, Tabel 8: 0,5 / 0,5")
            b1, b2 = st.columns(2)
            w_sem = b1.number_input("Semantik", 0.0, 10.0, 0.5, 0.1, key="w_sem")
            w_vis = b2.number_input("Visi", 0.0, 10.0, 0.5, 0.1, key="w_vis")
        st.markdown("**Penalaran**")
        level2 = st.checkbox("Aturan tingkat-2 (ekstensi implementasi ini)", value=True)

    b_go, b_last = st.columns([3, 1])
    go = b_go.button(GO, type="primary", width="stretch", icon=":material/search:", key="go")
    reopen = b_last.button("Lihat hasil terakhir", icon=":material/history:", width="stretch",
                           disabled="snap" not in st.session_state)
    msg_area = st.container()
    if "snap" not in st.session_state:
        msg_area.info(f"Unggah foto daun dan/atau isi kondisi lahan, lalu tekan **{GO}**. "
                      "Hasil akan muncul sebagai jendela pop-up.")

if model_note is not None and not vision_ready:
    model_note.caption("⚠️ Model visi belum siap. Buka **Pengaturan lanjutan** di bawah untuk melihat "
                       "penyebabnya, atau pilih **Tanpa foto**.")

# ================================================================== proses
if go:
    try:
        if not use_sensors and vision_source == VS_NONE:
            raise ValueError("Pilih minimal satu sumber: data kondisi lahan atau foto daun.")
        snap = {"diag": None, "vision": None, "photo": None, "vision_source": vision_source}
        if use_sensors:
            snap["values"], snap["rainy"] = dict(vals), bool(rainy)
            snap["diag"] = run_inference(tuple(sorted(vals.items())), rainy, level2)
        if vision_source == VS_PHOTO:
            if photo is None:
                raise ValueError("Unggah foto daun terlebih dahulu, atau pilih **Tanpa foto**.")
            if not vision_ready:
                raise ValueError("Model visi belum siap (lihat Pengaturan lanjutan).")
            data = photo.getvalue()
            snap["vision"], snap["photo"] = predict_image(data, ckpt_path, ckpt_mtime), data
        elif vision_source == VS_MANUAL:
            snap["vision"] = dict(vision_vec_manual)
        st.session_state["snap"] = snap
    except ValueError as exc:
        msg_area.error(str(exc))
        go = False
    except Exception as exc:  # pragma: no cover - tampilan galat
        msg_area.error(f"Inferensi gagal: {exc}. Pastikan Java terpasang (`java -version`) dan checkpoint valid.")
        go = False

snap = st.session_state.get("snap")
out = res = None
if snap is not None:
    res = snap["diag"]
    out = DecisionEngine(level2=level2).predict(diagnosis=res, vision_vector=snap["vision"],
                                                 w_semantic=w_sem, w_vision=w_vis)
    if go or reopen:
        show_result(out, res, snap, w_sem, w_vis)

# ================================================================== PUSTAKA PENYAKIT
with tab_lib:
    st.markdown("### 📚 Pustaka penyakit singkong")
    st.caption("Kenali ciri, penyebab, dan cara pengendalian tiap kelas yang dapat dikenali sistem.")
    lc = st.columns(len(CLASSES))
    for i, c in enumerate(CLASSES):
        inf, s = DISEASE_INFO[c], STYLE[c]
        short = inf.definition if len(inf.definition) <= 140 else inf.definition[:137].rsplit(" ", 1)[0] + " ..."
        lc[i].markdown(
            f'<div class="lib" style="--c:{s["color"]};--ct:{tint(s["color"], "1A")}"><div class="ic">{s["icon"]}</div>'
            f'{pill(c, s["color"])}<div class="t">{esc(inf.name_id)}</div>'
            f'<div class="muted"><i>{esc(inf.name_en)}</i></div>'
            f'<div class="d">{esc(short)}</div></div>', unsafe_allow_html=True)
    st.write("")
    pick = st.radio("Lihat detail", list(CLASSES), horizontal=True, key="lib_pick",
                    format_func=lambda c: f"{STYLE[c]['icon']} {DISEASE_INFO[c].name_id}")
    render_info(DISEASE_INFO[pick])

# ================================================================== MODE PAKAR
with tab_expert:
    st.markdown("### 🧠 Mode pakar: penalaran yang dapat ditelusuri")
    st.caption("Isi mesin inferensi: jejak aturan SWRL, fakta baru hasil Pellet, knowledge graph "
               "(T-Box & A-Box), serta basis aturan dan competency questions (CQ).")
    e_trace, e_facts, e_graph, e_kb = st.tabs(
        ["Jejak aturan", "Fakta turunan", "Knowledge graph (T-Box & A-Box)", "Basis pengetahuan & CQ"])

    with e_trace:
        if res is None:
            st.info("Jejak aturan tersedia setelah diagnosa dijalankan dengan data kondisi lahan.")
        else:
            st.markdown("Setiap baris = satu aturan IF–THEN. Kolom **Dipicu** berasal dari reasoner Pellet; "
                        "kolom **Nilai teramati** dan **Cocok dgn Python** adalah verifikasi independen.")
            rows = []
            for code in ("CBB", "CBSD", "CGM", "CMD"):
                for t in res.diseases[code].traces:
                    rows.append({
                        "Aturan": t.rule_id, "Penyakit": code, "JIKA": pretty_cond(t.if_text),
                        "Nilai teramati": fmt_obs(t.observed),
                        "Dipicu": "✅ ya" if t.fired else "— tidak",
                        "Cocok dgn Python": "ya" if t.python_agrees else "TIDAK",
                        "Arti kondisi (T-Box)": t.definition,
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            with st.expander("Bentuk SWRL tiap aturan"):
                st.code("\n".join(f"{r.rule_id}: {r.swrl()}" for r in RULES), language="text")

    with e_facts:
        if res is None:
            st.info("Fakta turunan tersedia setelah diagnosa dijalankan dengan data kondisi lahan.")
        else:
            st.markdown("Fakta **baru** yang dihasilkan Pellet dari A-Box + aturan (tidak ada di data masukan):")
            st.dataframe(pd.DataFrame(res.derived, columns=["Subjek", "Predikat", "Objek"]),
                         hide_index=True, width="stretch")
            if level2:
                st.caption("`isFullyFavourableFor` dan tipe `HighRiskPlant` berasal dari aturan tingkat-2 + "
                           "klasifikasi DL (ekstensi implementasi ini): tanaman memenuhi SEMUA kondisi suatu "
                           "penyakit. Itu bukan diagnosis infeksi.")
            st.caption(f"Waktu reasoning: {res.seconds:.2f} s")

    with e_graph:
        st.markdown("Ontologi dipisah menjadi **T-Box** (skema: kelas, hierarki, relasi, individu "
                    "pengetahuan pakar) dan **A-Box** (fakta tentang tanaman yang sedang didiagnosa). "
                    "Seret simpul untuk menata, gulir untuk memperbesar, arahkan kursor untuk detail.")
        kg_view = st.segmented_control(
            "Lapisan ontologi", ["T-Box (skema)", "A-Box (fakta)"], default="T-Box (skema)",
            required=True, key="kg_view", label_visibility="collapsed",
            format_func=lambda v: (":material/schema: " if v.startswith("T") else ":material/hub: ") + v)
        if kg_view.startswith("T"):
            summ = kgraph.tbox_summary()
            with st.container(horizontal=True):
                for k_, v_ in summ.items():
                    st.metric(k_, v_, border=True)
            show_ind = st.toggle("Tampilkan individu T-Box (penyakit, kondisi, gejala, properti teramati)",
                                 key="kg_tbox_ind")
            kg_t = kgraph.tbox_graph(include_individuals=show_ind)
            st.echarts_chart(kgraph.to_echarts(kg_t, kgraph.TBOX_CATEGORIES), height=620 if show_ind else 520,
                             key="kg_tbox_chart", alt="Graf T-Box ontologi penyakit singkong")
            st.caption("Hijau = kelas ontologi paper · biru tua = kelas SOSA (W3C) · ungu = kelas ekstensi "
                       "implementasi ini. Panah berlabel = object property (domain → range) atau subClassOf.")
            with st.expander("Tripel T-Box", icon=":material/table_rows:"):
                st.dataframe(pd.DataFrame(kg_t.triples()), hide_index=True, width="stretch")
        elif res is None or not snap.get("values"):
            st.info("A-Box terisi setelah diagnosa dijalankan dengan data kondisi lahan.",
                    icon=":material/hub:")
        else:
            show_types = st.toggle("Tampilkan tipe (rdf:type) individu", value=False, key="kg_abox_types")
            kg_a = kgraph.abox_graph(snap["values"], snap["rainy"], res.derived, show_types=show_types)
            n_inf = sum(x["kind"] == kgraph.INFERRED for x in kg_a.links)
            with st.container(horizontal=True):
                st.metric("Individu & nilai", len(kg_a.nodes), border=True)
                st.metric("Fakta masukan", sum(x["kind"] == kgraph.ASSERTED for x in kg_a.links), border=True)
                st.metric("Fakta hasil inferensi", n_inf, border=True)
            st.echarts_chart(kgraph.to_echarts(kg_a, kgraph.ABOX_CATEGORIES), height=560,
                             key="kg_abox_chart", alt="Graf A-Box tanaman yang didiagnosa")
            st.caption("Garis abu-abu = fakta masukan (asserted) · garis merah putus-putus = fakta baru hasil "
                       "reasoner Pellet (inferred) · garis `enablesDisease` berasal dari T-Box.")
            with st.expander("Tripel A-Box", icon=":material/table_rows:"):
                st.dataframe(pd.DataFrame(kg_a.triples()), hide_index=True, width="stretch")

    with e_kb:
        st.markdown("**Basis aturan (14 aturan SWRL tingkat-1)**")
        st.dataframe(pd.DataFrame([{
            "Aturan": r.rule_id, "Penyakit": r.disease, "JIKA": pretty_cond(r.condition_text()),
            "MAKA (hasCondition)": r.condition, "Sumber": r.source} for r in RULES]),
            hide_index=True, width="stretch")
        st.markdown("**Competency questions (SPARQL atas T-Box + A-Box)**")
        q = st.selectbox("Pilih CQ", [
            "CQ1 Penyakit apa saja yang mungkin?", "CQ2 Penyakit dengan gejala tertentu (kata kunci)",
            "CQ3 Hubungan tanaman–tanah–ladang–observasi", "CQ4 Gejala penyakit X",
            "CQ5 Penyakit karena virus/bakteri/jamur"])
        if q.startswith("CQ1"):
            st.write(cq.cq1())
        elif q.startswith("CQ2"):
            kw = st.text_input("Kata kunci gejala (bahasa Inggris, sesuai T-Box)", "mosaic")
            st.dataframe(pd.DataFrame(cq.cq2(kw), columns=["Penyakit", "Teks gejala"]), hide_index=True)
        elif q.startswith("CQ3"):
            st.dataframe(pd.DataFrame(cq.cq3(), columns=["Tanaman", "Tanah", "Ladang", "Musim hujan"]),
                         hide_index=True)
            st.dataframe(pd.DataFrame(cq.cq3_observations(), columns=["Objek teramati", "Properti", "Nilai"]),
                         hide_index=True)
        elif q.startswith("CQ4"):
            names = {"CBB": "CassavaBacterialBlight", "CBSD": "CassavaBrownStreakDisease",
                     "CGM": "CassavaGreenMottle", "CMD": "CassavaMosaicDisease"}
            pick4 = st.selectbox("Penyakit", list(names), key="cq4")
            st.write(cq.cq4(names[pick4]))
        else:
            kind = st.selectbox("Jenis", ["Viral", "Bacterial", "Fungal"])
            cq5 = cq.cq5(kind)
            st.write(cq5 if cq5 else "Tidak ada individu penyakit bertipe ini pada T-Box paper.")

# ================================================================== TENTANG
with tab_about:
    a1, a2 = st.columns([3, 2], gap="large")
    with a1:
        st.markdown("### Cara kerja sistem")
        st.markdown("Sistem ini mereproduksi Chhetri et al. (2023), *Expert Systems with Applications* "
                    "233:120955, dengan tiga komponen:")
        html_block(
            '<div class="act"><div class="n">1</div><div><b>Model visi</b><br><span class="muted">Foto daun '
            'diklasifikasikan oleh model <code>timm</code> (default VOLO-D2 384) menjadi vektor '
            'probabilitas 5 kelas.</span></div></div>'
            '<div class="act"><div class="n">2</div><div><b>Model semantik</b><br><span class="muted">Data lahan '
            'dimasukkan ke ontologi OWL; 14 aturan SWRL dinalar oleh Pellet menjadi skor kecocokan tiap penyakit '
            'beserta penjelasannya.</span></div></div>'
            '<div class="act"><div class="n">3</div><div><b>Decision engine</b><br><span class="muted">Kedua vektor '
            'digabung dengan weighted majority voting (Eq. 1), lalu info penyakit dan saran pengendalian '
            'ditampilkan.</span></div></div>')
        st.markdown("### Keterbatasan")
        st.markdown(
            "- Ambang aturan berasal dari Tabel 1 paper; data lahan pada demo adalah simulasi/masukan pengguna.\n"
            "- Skor semantik adalah rasio kondisi terpenuhi, bukan probabilitas terkalibrasi.\n"
            "- Model visi dilatih pada foto daun dari Uganda (iCassava), sehingga kinerja pada foto dari daerah "
            "atau varietas lain belum teruji.\n"
            "- Hasil **bukan diagnosis resmi**. Detail ada di README.md.")
    with a2:
        st.markdown("### Kinerja model visi")
        m = vision_metrics()
        if m and "final_test_metrics" in m:
            fm = m["final_test_metrics"]
            args = (m.get("meta") or {}).get("args") or {}
            k1, k2 = st.columns(2)
            k1.metric("Akurasi uji", f"{fm['accuracy'] * 100:.1f}%")
            k2.metric("Jumlah foto uji", f"{fm.get('n', 0):,}".replace(",", "."))
            st.caption(f"Arsitektur: {args.get('model', '-')} @ {args.get('img_size', '-')} px")
            st.dataframe(pd.DataFrame([
                {"Kelas": c, "Presisi": round(v["precision"], 3), "Recall": round(v["recall"], 3),
                 "F1": round(v["f1"], 3), "Jumlah": v["support"]}
                for c, v in fm["per_class"].items()]), hide_index=True, width="stretch")
        else:
            st.caption("Metrik model visi belum tersedia (notebooks/output/vision/vision_metrics.json).")

html_block('<div class="foot">Sistem Pakar Penyakit Singkong · reproduksi Chhetri et al. (2023) · '
           'untuk edukasi dan penelitian</div>')
