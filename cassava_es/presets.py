"""Contoh masukan untuk aplikasi (data ilustrasi, bukan pengukuran lapangan).

Setiap contoh dipilih agar kelas yang dinamai memiliki skor semantik tertinggi
SECARA UNIK (diuji di tests/test_presets.py), sehingga argmax tidak jatuh ke
kelas lain karena seri.
"""

PRESETS = {
    "Manual (atur sendiri)": None,
    "Contoh CBB (semua kondisi CBB terpenuhi)": dict(rainy=True, SoilMoisture=0.50, SoilPH=6.8, SoilTemperature=27.0, RelativeHumidity=0.60, Temperature=25.0),
    "Contoh CBSD": dict(rainy=False, SoilMoisture=0.20, SoilPH=6.0, SoilTemperature=20.0, RelativeHumidity=0.80, Temperature=20.0),
    "Contoh CGM": dict(rainy=False, SoilMoisture=0.80, SoilPH=6.0, SoilTemperature=30.0, RelativeHumidity=0.90, Temperature=25.0),
    "Contoh CMD": dict(rainy=False, SoilMoisture=0.40, SoilPH=6.0, SoilTemperature=25.0, RelativeHumidity=0.90, Temperature=33.0),
    "Contoh tanpa kondisi pendukung": dict(rainy=False, SoilMoisture=0.05, SoilPH=5.0, SoilTemperature=8.0, RelativeHumidity=0.30, Temperature=15.0),
}

EXPECTED_TOP = {
    "Contoh CBB (semua kondisi CBB terpenuhi)": "CBB",
    "Contoh CBSD": "CBSD",
    "Contoh CGM": "CGM",
    "Contoh CMD": "CMD",
    "Contoh tanpa kondisi pendukung": "Healthy",
}
