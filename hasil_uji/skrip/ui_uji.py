"""Uji antarmuka di peramban nyata (Chrome) dengan foto singkong nyata.

Pemakaian: python ui_uji.py URL DIR_DATA CSV_PREDIKSI DIR_KELUAR
"""
import csv
import re
import sys
import time

from playwright.sync_api import sync_playwright

url, data, pred_csv, out = sys.argv[1:5]
rows = list(csv.DictReader(open(pred_csv)))
order = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
# foto pertama per kelas menurut urutan nama (bukan dipilih berdasarkan benar/salah)
picks = {c: sorted((r for r in rows if r["benar"] == c), key=lambda r: r["image_id"])[0] for c in order}
log, problems = [], []


def say(*a):
    line = " ".join(str(x) for x in a)
    log.append(line)
    print(line, flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])

    def idle(pg, s=1.5):
        time.sleep(0.8)
        try:
            pg.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120000)
        except Exception:
            pass
        time.sleep(s)

    def start():
        pg = browser.new_page(viewport={"width": 1366, "height": 1100})
        pg.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        pg.goto(url)
        pg.get_by_text("Diagnosa sekarang").first.wait_for(timeout=90000)
        idle(pg)
        return pg

    def no_exception(pg, name):
        n = pg.locator('[data-testid="stException"]').count()
        if n:
            problems.append(f"{name}: {n} kotak exception")

    def go(pg):
        pg.get_by_role("button", name="Diagnosa sekarang").click()
        pg.get_by_role("dialog").wait_for(timeout=120000)
        idle(pg, 2.5)
        return pg.get_by_role("dialog")

    def set_number(pg, label, value):
        box = pg.get_by_label(label, exact=True)
        box.fill(str(value))
        box.press("Enter")
        idle(pg, 0.6)

    def preset(pg, name):
        pg.locator('[data-testid="stSelectbox"]').first.click()
        pg.get_by_role("option", name=name).click()
        idle(pg)

    # ---- 1. foto saja, satu foto per kelas
    for i, c in enumerate(order, 1):
        pg = start()
        pg.get_by_text("Pakai data kondisi lahan (sensor)").click()
        idle(pg)
        r = picks[c]
        pg.locator('input[type="file"]').first.set_input_files(f"{data}/{r['image_id']}")
        idle(pg, 2.5)
        text = go(pg).inner_text()
        head = re.search(r"(Kesimpulan: [^\n]+|Hasil belum konklusif[^\n]*)", text)
        expected = r["prediksi"]
        ok = head is not None and (f"({expected})" in head[0] or (expected == "Healthy" and "Sehat" in head[0]))
        say(f"foto {r['image_id']} | label dataset {c} | prediksi CSV {expected} | layar: "
            f"{head[0] if head else '-'} | cocok dengan CSV: {ok}")
        if not ok:
            problems.append(f"hasil layar tidak cocok untuk {r['image_id']}")
        pg.screenshot(path=f"{out}/0{i}_foto_saja_{c}.png")
        no_exception(pg, f"foto {c}")
        pg.close()

    # ---- 2. fusi foto CMD + contoh data CMD
    pg = start()
    preset(pg, "Contoh CMD")
    pg.locator('input[type="file"]').first.set_input_files(f"{data}/{picks['CMD']['image_id']}")
    idle(pg, 2.5)
    d = go(pg)
    text = d.inner_text()
    say("fusi CMD:", re.search(r"(Kesimpulan: [^\n]+|Hasil belum konklusif[^\n]*)", text)[0],
        "| teks Eq.1:", "Weighted majority voting" in text, "| catatan R14:", "CMDHighRainfall" in text)
    pg.screenshot(path=f"{out}/06_fusi_cmd.png")
    d.get_by_role("tab", name="Rincian angka").click()
    idle(pg, 2)
    pg.screenshot(path=f"{out}/07_fusi_cmd_rincian.png")
    no_exception(pg, "fusi")
    pg.close()

    # ---- 3. skor seri dari data lahan
    pg = start()
    pg.get_by_text("Tanpa foto").click()
    idle(pg)
    pg.get_by_text("Sedang musim hujan di ladang").click()
    idle(pg)
    for label, v in (("💧 Kelembapan tanah (0-1)", 0.8), ("🧪 pH tanah (0-14)", 6.8),
                     ("🌡️ Suhu tanah (°C)", 28), ("☁️ Kelembapan udara (0-1)", 0.9),
                     ("☀️ Suhu udara (°C)", 25)):
        set_number(pg, label, v)
    d = go(pg)
    text = d.inner_text()
    say("seri data lahan:", "belum konklusif" in text, "| CBB:", "(CBB)" in text, "| CGM:", "(CGM)" in text)
    if "belum konklusif" not in text:
        problems.append("skor seri tidak ditampilkan")
    pg.screenshot(path=f"{out}/08_seri.png")
    d.get_by_role("tab", name="Mengapa hasil ini?").click()
    idle(pg, 2)
    pg.screenshot(path=f"{out}/09_seri_alasan.png")
    no_exception(pg, "seri")
    pg.close()

    # ---- 4. bobot nol (sebelumnya membuat aplikasi crash)
    pg = start()
    pg.get_by_text("Vektor manual (pengujian)").click()
    idle(pg)
    pg.get_by_text("Pengaturan lanjutan").click()
    idle(pg)
    set_number(pg, "Semantik", 0)
    set_number(pg, "Visi", 0)
    pg.get_by_role("button", name="Diagnosa sekarang").click()
    idle(pg, 4)
    alert = pg.get_by_text("tidak boleh nol keduanya")
    say("bobot nol: pesan tampil:", alert.count() > 0, "| dialog:", pg.get_by_role("dialog").count())
    if not alert.count():
        problems.append("pesan bobot nol tidak tampil")
    else:
        alert.first.scroll_into_view_if_needed()
    pg.screenshot(path=f"{out}/10_bobot_nol.png")
    no_exception(pg, "bobot nol")
    pg.close()

    # ---- 5. vektor nol semua
    pg = start()
    pg.get_by_text("Pakai data kondisi lahan (sensor)").click()
    idle(pg)
    pg.get_by_text("Vektor manual (pengujian)").click()
    idle(pg)
    for c in order:
        set_number(pg, f"p_visi({c})", 0)
    pg.get_by_role("button", name="Diagnosa sekarang").click()
    idle(pg, 4)
    alert = pg.get_by_text("bernilai nol semua")
    say("vektor nol: pesan tampil:", alert.count() > 0)
    if not alert.count():
        problems.append("pesan vektor nol tidak tampil")
    else:
        alert.first.scroll_into_view_if_needed()
    pg.screenshot(path=f"{out}/11_vektor_nol.png")
    no_exception(pg, "vektor nol")
    pg.close()

    # ---- 6. mode pakar: catatan R14 dan CGM; tab Tentang
    pg = start()
    preset(pg, "Contoh CMD")
    pg.get_by_text("Tanpa foto").click()
    idle(pg)
    go(pg)
    pg.keyboard.press("Escape")
    idle(pg)
    pg.get_by_role("tab", name="Mode pakar (XAI)").click()
    idle(pg, 2)
    note = pg.get_by_text("Catatan R14").first
    say("catatan R14 di jejak aturan:", note.is_visible())
    note.scroll_into_view_if_needed()
    pg.screenshot(path=f"{out}/12_pakar_r14.png")
    pg.get_by_role("tab", name="Basis pengetahuan & CQ").click()
    idle(pg)
    box = pg.locator('[data-testid="stSelectbox"]').filter(has_text="Pilih CQ")
    box.scroll_into_view_if_needed()
    box.locator("input").click()
    idle(pg, 0.5)
    box.locator("input").fill("CQ5")
    idle(pg, 0.8)
    pg.keyboard.press("Enter")
    idle(pg, 2)
    note = pg.get_by_text("Catatan nama CGM").first
    say("catatan CGM di CQ5:", note.is_visible())
    note.scroll_into_view_if_needed()
    pg.screenshot(path=f"{out}/13_cq5_cgm.png")
    pg.get_by_role("tab", name="Tentang").first.click()
    idle(pg, 2)
    pg.get_by_text("Angka ini dibaca dari").first.scroll_into_view_if_needed()
    pg.screenshot(path=f"{out}/14_tentang.png")
    no_exception(pg, "mode pakar")
    pg.close()
    browser.close()

say("MASALAH:", problems or "tidak ada")
open(f"{out}/log_uji_antarmuka.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
sys.exit(1 if problems else 0)
