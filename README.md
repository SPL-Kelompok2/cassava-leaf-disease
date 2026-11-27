# Sistem Pakar Penyakit Singkong

Sistem pakar untuk mendeteksi penyakit dan hama singkong dengan menggabungkan **ontologi OWL + aturan SWRL + penalaran Pellet**
(model semantik), **klasifikasi foto daun** (model visi), dan **decision engine** yang menyatukan keduanya.
Setiap hasil dilengkapi penjelasan berbasis aturan dan info umum tentang penyakit yang diprediksi.

Implementasi mengacu pada:

> Chhetri, T. R., Hohenegger, A., Fensel, A., Kasali, M. A., & Adekunle, A. A. (2023).
> *Towards improving prediction accuracy and user-level explainability using deep learning and knowledge graphs:
> A study on cassava disease.* Expert Systems With Applications, 233, 120955.
> https://doi.org/10.1016/j.eswa.2023.120955

> Data sensor pada demo adalah simulasi atau masukan pengguna. Keluaran adalah skor kesesuaian kondisi dan prediksi
> model, **bukan diagnosis resmi**.

## Fitur

- **Model semantik**: 14 aturan SWRL (ambang Tabel 1 paper) dijalankan oleh Pellet atas ontologi dan data sensor
  (SOSA); keluarannya vektor skor per kelas, fakta turunan, dan jejak aturan.
- **Model visi**: klasifikasi 5 kelas (CBB, CBSD, CGM, CMD, Healthy) dengan arsitektur apa pun dari `timm`
  (default VOLO-D2 384), lengkap dengan skrip pelatihan dan notebook Kaggle.
- **Decision engine**: weighted majority voting (Eq. 1 paper) atas citra + sensor, dengan mode gabungan, semantik saja,
  atau visi saja; keluaran JSON.
- **Info penyakit**: definisi, penyebab, gejala, penyebaran, pengendalian umum, dan sumber untuk tiap kelas.
- **Aplikasi Streamlit**: input sensor, unggah foto, bobot penggabungan, penjelasan, jejak aturan, fakta turunan,
  dan competency questions (SPARQL).
- Model visi bersifat opsional; tanpa torch aplikasi berjalan sebagai sistem pakar murni.

## Arsitektur

```
foto daun ──> model visi ─────┐
                              ├─> weighted majority voting (Eq. 1) ─> kelas akhir ─> penjelasan + info penyakit
data sensor ─> SWRL + Pellet ─┘
```

## Struktur repositori

```
cassava-expert-system/
├── app.py                         aplikasi Streamlit
├── requirements.txt               dependensi inti
├── requirements-vision.txt        torch, torchvision, timm (opsional, untuk model visi)
├── packages.txt                   paket sistem (Java) untuk Streamlit Community Cloud
├── cassava_es/
│   ├── rules.py                   tabel 14 aturan -> teks SWRL
│   ├── kb.py                      muat T-Box, isi A-Box, jalankan Pellet, hitung vektor dan jejak aturan
│   ├── decision.py                weighted majority voting (Eq. 1)
│   ├── engine.py                  decision engine: citra + sensor -> keputusan + penjelasan + info
│   ├── vision.py                  muat checkpoint, foto -> probabilitas 5 kelas
│   ├── losses.py                  Taylor Cross Entropy + label smoothing
│   ├── data.py                    baca merged.csv, pemetaan label, split, metrik
│   ├── disease_info.py            info umum tiap penyakit
│   ├── simulate.py                simulator data sensor (Tabel 1)
│   ├── presets.py                 contoh masukan untuk aplikasi
│   └── cq.py                      competency questions (SPARQL)
├── semantic-model/ontology_service/
│   ├── owl/                       cassava-disease.owl (T-Box), cassava-extension.owl, cassava-rules.owl, sosa.owl, .ttl
│   └── abox/                      cassava-abox.owl / .ttl, cassava-abox-inferred.ttl
├── scripts/
│   ├── train_vision.py            melatih model visi dari merged.csv
│   ├── build_extension.py         membangun cassava-extension.owl
│   └── build_abox.py              membangun A-Box demo dan berkas Turtle
├── notebooks/
│   └── train_vision_kaggle.ipynb  pelatihan model visi di Kaggle (GPU)
└── tests/                         136 tes pytest
```

## Instalasi

Prasyarat: **Java** (untuk Pellet) dan Python >= 3.10.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Model visi (opsional):

```powershell
pip install -r requirements-vision.txt
```

Untuk GPU NVIDIA, pasang build CUDA dari https://pytorch.org/get-started/locally/ sebelum perintah di atas.

Catatan kompatibilitas: `owlready2` harus **0.37**. Versi yang lebih baru membawa Pellet yang dikompilasi untuk Java yang
lebih baru dan gagal pada Java 21 (`UnsupportedClassVersionError`).

## Menjalankan aplikasi

```powershell
streamlit run app.py        # http://localhost:8501
```

1. Pilih contoh data atau isi nilai sensor di panel kiri (centang/hapus *Pakai data sensor*).
2. Pilih sumber visi: *Tidak dipakai*, *Foto daun (model terlatih)*, atau *Vektor probabilitas manual*.
   Untuk foto, isi path checkpoint (default `checkpoints/vision.pt`, atau variabel lingkungan `CASSAVA_VISION_CKPT`) dan unggah foto.
3. Atur bobot model semantik dan model visi bila keduanya dipakai (default 0,5 / 0,5).
4. Tekan **Jalankan inferensi**, lalu baca tab *Hasil & penjelasan*, *Jejak aturan (XAI)*, *Fakta turunan*, dan
   *Basis pengetahuan & CQ*.

Satu pemanggilan Pellet memakan sekitar 1-2 detik; hasil disimpan sementara agar masukan yang sama tidak diproses ulang.
Di Streamlit Community Cloud, `packages.txt` sudah memuat `default-jre-headless`.

## Model semantik

### Basis pengetahuan

**T-Box** (`owl/cassava-disease.owl`): 16 kelas bernama, 12 properti objek, 1 properti data (`hasRainySeason`), 27 individu.

- Hierarki penyakit: `PlantDisease` ⊃ `BacterialPlantDisease`, `ViralPlantDisease`, `FungalPlantDisease`.
  Individu: CBB (bakteri); CBSD, CGM, CMD (virus).
- Konteks tanam: `Field` -`hasSoil`-> `Soil` -`hasPlant`-> `Plant` (dan inversnya `isPartOfField`, `isPlantedIn`).
- Kondisi: kelas `DiseaseCondition` (+ `CBBCondition`, `CBSDCondition`, `CGMCondition`, `CMDCondition`); 14 individu,
  masing-masing terhubung ke penyakitnya lewat `enablesDisease` dan punya anotasi `skos:definition` (teks penjelasan).
  Hasil inferensi disimpan sebagai `hasCondition(Plant, DiseaseCondition)`.
- Gejala: 4 individu `DiseaseSymptom` (satu per penyakit, teks bebas) lewat `hasSymptoms`/`isCausedByDisease`.
- Sensor: SOSA (`sosa:Observation`, `observedProperty`, `hasSimpleResult`, `hasFeatureOfInterest`) dengan 5
  `sosa:ObservableProperty`: SoilMoisture, SoilPH, SoilTemperature, RelativeHumidity, Temperature.

**A-Box** (`abox/cassava-abox.owl` / `.ttl`): 5 ladang, 5 tanah, 5 tanaman, dan 25 observasi SOSA (5 per tanaman),
satu tanaman per kelas. Nilai sensor disimulasikan dari Tabel 1 dengan seed tetap (2023) lewat `scripts/build_abox.py`.
`cassava-abox-inferred.ttl` berisi A-Box ditambah fakta hasil inferensi Pellet. Namespace `http://example.org/cassava-abox#`
adalah placeholder.

### Aturan IF-THEN

Setiap aturan berbentuk: kondisi tanaman terpenuhi bila observasi terkait memenuhi ambang, lalu `hasCondition`.
Contoh bentuk SWRL (R04):

```
Plant(?plant), isPlantedIn(?plant, ?soil), isFeatureOfInterestOf(?soil, ?obs),
observedProperty(?obs, SoilTemperature), hasSimpleResult(?obs, ?result),
greaterThanOrEqual(?result, 25), lessThanOrEqual(?result, 30)
-> hasCondition(?plant, CBBSoilTemperature)
```

Sensor tanah menempel pada `Soil`; kelembapan relatif, suhu udara, dan musim hujan menempel pada `Field`.
Batas bersifat inklusif (>=, <=).

| ID | Penyakit | JIKA | MAKA |
|---|---|---|---|
| R01 | CBB | musim hujan = ya | `hasCondition(?plant, CBBRainySeason)` |
| R02 | CBB | kelembapan tanah >= 0.3 (0-1) | `hasCondition(?plant, CBBSoilMoisture)` |
| R03 | CBB | 6.5 <= pH tanah <= 7.2 | `hasCondition(?plant, CBBSoilPH)` |
| R04 | CBB | 25 <= suhu tanah <= 30 °C | `hasCondition(?plant, CBBSoilTemperature)` |
| R05 | CBSD | 0.7 <= kelembapan relatif udara <= 0.85 | `hasCondition(?plant, CBSDRelativeHumidity)` |
| R06 | CBSD | kelembapan tanah >= 0.1 | `hasCondition(?plant, CBSDSoilMoisture)` |
| R07 | CBSD | 10 <= suhu tanah <= 32 °C | `hasCondition(?plant, CBSDSoilTemperature)` |
| R08 | CGM | kelembapan tanah >= 0.7 | `hasCondition(?plant, CGMSoilMoisture)` |
| R09 | CGM | kelembapan relatif udara >= 0.7 | `hasCondition(?plant, CGMRelativeHumidity)` |
| R10 | CGM | suhu tanah >= 27 °C | `hasCondition(?plant, CGMSoilTemperature)` |
| R11 | CMD | suhu udara >= 30 °C | `hasCondition(?plant, CMDTemperature)` |
| R12 | CMD | kelembapan tanah >= 0.3 | `hasCondition(?plant, CMDSoilMoisture)` |
| R13 | CMD | 20 <= suhu tanah <= 32 °C | `hasCondition(?plant, CMDSoilTemperature)` |
| R14 | CMD | kelembapan relatif udara >= 0.8 | `hasCondition(?plant, CMDHighRainfall)` |

Sumber ambang: R02-R14 diambil dari kolom *Lower/Upper* Tabel 1 paper; R01 berasal dari individu `CBBRainySeason`
pada ontologi. Tabel aturan di `cassava_es/rules.py` adalah satu-satunya sumber; teks SWRL dihasilkan darinya.
Aturan juga tersedia sebagai OWL di `owl/cassava-rules.owl`.

### Penalaran dan skor

1. T-Box dan SOSA dimuat lokal (SOSA lebih dulu agar `owl:imports` tidak mengunduh dari web).
2. Data sensor menjadi individu A-Box (`sosa:Observation`).
3. Aturan SWRL dipasang dan `sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)` dijalankan.
4. Fakta baru `hasCondition(plant, kondisi)` dibaca dari hasil Pellet.
5. Skor semantik:
   - `p(penyakit) = (kondisi penyakit tsb yang terpenuhi) / (jumlah kondisi penyakit tsb)`
   - `p(Healthy) = 1 - (seluruh kondisi terpenuhi) / 14`

Contoh: musim hujan, kelembapan tanah 0,5, pH 6,8, suhu tanah 27, RH 0,6, suhu udara 25 → CBB 4/4 = 1,0; CBSD 2/3 = 0,667;
CGM 1/3 = 0,333; CMD 2/4 = 0,5; Healthy = 1 - 9/14 = 0,357.

Skor adalah rasio kondisi terpenuhi, bukan probabilitas terkalibrasi; beberapa penyakit dapat bernilai 1,0 bersamaan.
Bila skor seri, argmax memilih kelas pertama menurut urutan CBB, CBSD, CGM, CMD, Healthy, dan aplikasi menampilkan peringatan.

**Penggabungan (Eq. 1).** `argmax_c Σ w_i p_ic` dengan `i ∈ {visi, semantik}`. Hasil dihitung sebagai rata-rata berbobot
(dibagi jumlah bobot), sehingga contoh angka paper (bobot 0,3/0,8 → [0,414; 0,232; 0,305; 0,415; 0,275]) tereproduksi.

### Penjelasan (XAI)

- **Definisi kondisi**: `skos:definition` tiap kondisi yang terpenuhi, ditambah teks gejala penyakit.
- **Jejak aturan**: untuk semua 14 aturan ditampilkan ID, bagian JIKA, nilai teramati, status menyala menurut Pellet, dan
  pencocokan dengan evaluasi Python independen.

### Ekstensi

| Ekstensi | Penjelasan |
|---|---|
| Aturan tingkat-2 | 4 aturan SWRL berantai: bila semua kondisi suatu penyakit terpenuhi → `isFullyFavourableFor(plant, penyakit)`. Tanpa ambang baru; dapat dimatikan di aplikasi |
| `HighRiskPlant` | Kelas terdefinisi `Plant ⊓ ∃isFullyFavourableFor.PlantDisease`, diklasifikasikan Pellet. Ada di `cassava-extension.owl`, bukan di T-Box |
| Jejak aturan | Lihat bagian Penjelasan |
| Competency questions | CQ1-CQ5 dijawab dengan SPARQL (`cassava_es/cq.py`); CQ2 berupa pencarian kata kunci pada teks gejala |
| Ekspor Turtle | T-Box, A-Box, dan A-Box hasil inferensi dalam `.ttl` |
| Isolasi per-permintaan | Tiap diagnosis memakai "dunia" owlready2 baru |

Membangun ulang artefak: `python scripts/build_extension.py` lalu `python scripts/build_abox.py`.

## Model visi

### Format dataset

Satu folder citra datar (tanpa sub-folder per penyakit) dan satu `merged.csv` dengan kolom `image_id, label, source`.

- `label`: nama (`cbb`, `CMD`, `healthy`, ...) atau angka 0-4 (0=CBB, 1=CBSD, 2=CGM, 3=CMD, 4=Healthy).
- `image_id`: boleh dengan atau tanpa ekstensi.
- Dataset yang dipakai: `tahsin/cassava-leaf-disease-merged` di Kaggle (gabungan iCassava 2019 dan 2020).

Skrip mencetak distribusi kelas berdampingan dengan distribusi pada paper (Gambar 7) untuk memeriksa pemetaan label;
label tak dikenal menghentikan proses, berkas yang hilang dibuang dan dilaporkan.

### Pelatihan

```powershell
python scripts/train_vision.py --csv data/merged.csv --img-dir data/train --out checkpoints/vision.pt
```

Default mengikuti paper (Bagian 4.5, Tabel 5): `volo_d2_384`, 384x384, 10 epoch, batch 16, Adam (lr 1e-4, weight decay 1e-6),
CosineAnnealingWarmRestarts (T_0=10, T_mult=1, eta_min=1e-6), Taylor CE + label smoothing 0,05, bobot ImageNet,
pembagian 80/20 terstratifikasi (seed 2023), dan augmentasi RandomResizedCrop, HorizontalFlip, Transpose, VerticalFlip,
ShiftScaleRotate. Opsi lain: `--model`, `--img-size`, `--epochs`, `--batch-size`, `--num-workers`, `--limit`, `--no-pretrained`;
untuk GPU berkapasitas kecil, contoh `--model tf_efficientnetv2_s --img-size 384`.

Keluaran di folder `--out`:

| Berkas | Isi |
|---|---|
| `vision.pt` | checkpoint epoch terakhir |
| `vision_metrics.json` | konfigurasi, laporan dataset, riwayat epoch, presisi/recall/F1, matriks konfusi |
| `vision_split.csv` | pembagian latih/uji |
| `vision_test_preds.csv` | probabilitas per citra uji |

Catatan implementasi:

- Checkpoint yang disimpan adalah epoch terakhir (tanpa pemilihan epoch berdasarkan data uji); akurasi uji per epoch hanya dicetak.
- Pra-pemrosesan evaluasi dan inferensi memakai fungsi yang sama (`vision.preprocess`: resize ke `img_size`x`img_size`
  + normalisasi ImageNet); orientasi EXIF foto diterapkan.
- Checkpoint dimuat dengan `weights_only=True`.

### Notebook Kaggle

`notebooks/train_vision_kaggle.ipynb` menjalankan pelatihan di GPU Kaggle. Langkah:

1. Session options: aktifkan GPU dan Internet.
2. Add Input: dataset citra (`tahsin/cassava-leaf-disease-merged`) dan dataset kode (unggah zip repositori sebagai Dataset).
3. Jalankan sel berurutan; sel *uji cepat* (300 citra, 1 epoch) mengukur waktu sebelum pelatihan penuh.
4. Setelah selesai, **Save Version** lalu unduh `vision/vision.pt` dari tab Output.
5. Letakkan `vision.pt` di `checkpoints/` dan pilih *Foto daun* di aplikasi.

Notebook mendeteksi lokasi kode, `merged.csv`, dan folder citra secara otomatis.

### Hasil pelatihan referensi

Konfigurasi default (VOLO-D2 384, 10 epoch, batch 16, AMP) pada GPU Kaggle. Dataset: 26.337 citra
(CBB 1.492, CBSD 3.476, CGM 3.017, CMD 15.462, Healthy 2.890), latih 21.071 / uji 5.266.

| Kelas | Presisi | Recall | F1 | n |
|---|---|---|---|---|
| CBB | 0,696 | 0,685 | 0,690 | 298 |
| CBSD | 0,864 | 0,847 | 0,855 | 695 |
| CGM | 0,874 | 0,851 | 0,862 | 603 |
| CMD | 0,959 | 0,973 | 0,966 | 3.092 |
| Healthy | 0,784 | 0,765 | 0,774 | 578 |

Akurasi uji (epoch terakhir): **0,9035**. Waktu: sekitar 1.105 detik per epoch, total sekitar 3,1 jam.

## Decision engine

```python
from cassava_es.engine import DecisionEngine
from cassava_es.vision import VisionModel

engine = DecisionEngine(vision=VisionModel.load("checkpoints/vision.pt"))
hasil = engine.predict(
    image="daun.jpg",
    sensors={"SoilMoisture": 0.5, "SoilPH": 6.8, "SoilTemperature": 27.0,
             "RelativeHumidity": 0.6, "Temperature": 25.0},
    rainy=True,
    w_semantic=0.5, w_vision=0.5,
)
print(hasil.label, hasil.visual_certainty, hasil.knowledge_certainty)
print(hasil.to_json(indent=2))
```

`FinalResult` berisi kelas akhir, vektor akhir, vektor semantik dan visi, `visual_certainty` (p visi kelas akhir),
`knowledge_certainty` (p semantik kelas akhir), penjelasan ontologi, info penyakit, dan `to_json()`.

| Mode | Masukan | Keterangan |
|---|---|---|
| `fusion` | citra/vektor visi + sensor | weighted majority voting (Eq. 1) |
| `semantic_only` | sensor | tanpa model visi |
| `vision_only` | citra/vektor visi | tanpa penjelasan ontologi (butuh sensor) |

Argumen `vision_vector` menerima vektor probabilitas visi yang sudah dihitung, dan `diagnosis` menerima hasil
`diagnose_one` agar Pellet tidak dijalankan ulang.

## Info penyakit

`cassava_es/disease_info.py` menyimpan info tiap kelas: definisi, penyebab, ciri yang tampak, cara menyebar, kondisi
pendukung, dampak, pengendalian umum, catatan, dan sumber. Informasi bersifat umum (tanpa merek atau dosis pestisida).
Kelas CGM pada dataset iCassava adalah *Cassava Green Mite* (tungau *Mononychellus tanajoa*), sehingga info CGM
mengikuti makna tersebut.

## Pengujian

```powershell
pytest -q        # 136 tes; tes yang memerlukan torch dilewati otomatis bila torch tidak terpasang
```

Cakupan: kecocokan ambang aturan dengan Tabel 1; kolom *Probability* Tabel 1 pada simulator; kesesuaian hasil Pellet dengan
evaluasi Python (acak dan nilai batas); contoh numerik Eq. (1) paper; checksum T-Box; kesetaraan `.ttl` dan `.owl`;
pemuatan `cassava-rules.owl`; pemetaan label dan metrik (dicocokkan dengan matriks konfusi Gambar 10 dan Tabel 7 paper);
loss Taylor CE; inferensi dan keamanan checkpoint; pelatihan ujung-ke-ujung pada data sintetis; decision engine; info penyakit;
dan aplikasi Streamlit (AppTest) termasuk jalur unggah foto.

## Keterbatasan

- Data sensor berupa simulasi atau masukan pengguna; ambang aturan bersumber dari Tabel 1 paper dan tidak divalidasi dengan data lapangan.
- Skor semantik adalah rasio aturan terpenuhi (bobot semua kondisi sama, tidak terkalibrasi); sensor yang tidak ada dianggap
  "tidak terpenuhi".
- Gejala pada ontologi berupa teks bebas; tidak ada penalaran berbasis gejala.
- Kelas CGM: ontologi memodelkannya sebagai penyakit virus ("Cassava Green Mottle"), sedangkan dataset iCassava dan literatur
  IITA menyebutnya tungau hijau (*Cassava Green Mite*). Aturan sensor CGM (R08-R10) mensyaratkan kelembapan tinggi,
  sedangkan IITA menyebut kerusakan lebih parah pada musim kering.
- Model visi dilatih pada foto daun dari Uganda (iCassava); kinerjanya pada foto dari daerah, varietas, atau kamera lain
  belum diuji. Satu foto daun tidak menunjukkan infeksi laten atau gejala di bagian lain (batang, umbi).
- Pada dataset, satu daun dapat mengandung lebih dari satu penyakit tetapi hanya memiliki satu label; kelas CBB, CGM, dan
  Healthy jauh lebih sedikit daripada CMD. Duplikat antar-tahun tidak disaring.
- Dataset citra tidak menyertakan data sensor, sehingga tidak ada evaluasi akurasi gabungan (visi + semantik) pada data nyata.

## Lisensi dan atribusi

- Paper: CC BY 4.0. T-Box (`cassava-disease.owl`) dan gagasan aturan berasal dari paper dan repositori penulis
  (https://github.com/Research-Tek/xai-cassava-agriculture).
- `sosa.owl`: ontologi SOSA/SSN, W3C.

## Referensi

- Chhetri, T. R., Hohenegger, A., Fensel, A., Kasali, M. A., & Adekunle, A. A. (2023). Expert Systems With Applications, 233, 120955.
- Janowicz, K., Haller, A., Cox, S. J., Le Phuoc, D., & Lefrançois, M. (2019). SOSA: A lightweight ontology for sensors, observations, samples, and actuators. *Journal of Web Semantics*, 56, 1-10.
- Horrocks, I., Patel-Schneider, P. F., Boley, H., Tabet, S., Grosof, B., & Dean, M. (2004). SWRL: A semantic web rule language combining OWL and RuleML. W3C Member Submission.
- Noy, N., & McGuinness, D. L. (2001). Ontology development 101. Stanford KSL-01-05.
- Jearanaiwongkul, W., Anutariya, C., & Andres, F. (2018). An ontology-based approach to plant disease identification system. *IAIT 2018*, ACM.
- Sirin, E., Parsia, B., Grau, B. C., Kalyanpur, A., & Katz, Y. (2007). Pellet: A practical OWL-DL reasoner. *Journal of Web Semantics*, 5(2), 51-53.
- Lamy, J.-B. (2017). Owlready: Ontology-oriented programming in Python. *Artificial Intelligence in Medicine*, 80, 11-28.
- Mwebaze, E., Gebru, T., Frome, A., Nsumba, S., & Tusubira, J. (2019). iCassava 2019 fine-grained visual categorization challenge. arXiv:1908.02900.
- Yuan, L., Hou, Q., Jiang, Z., Feng, J., & Yan, S. (2021). VOLO: Vision outlooker for visual recognition. arXiv:2106.13112.
- Feng, L., Shu, S., Lin, Z., Lv, F., Li, L., & An, B. (2021). Can cross entropy loss be robust to label noise? IJCAI.
- Zárate-Chaves, C. A., et al. (2021). Cassava diseases caused by *Xanthomonas phaseoli* pv. *manihotis* and *X. cassavae*. Molecular Plant Pathology, 22(12), 1520-1537.
- Chikoti, P. C., & Tembo, M. (2022). Expansion and impact of cassava brown streak and cassava mosaic diseases in Africa: a review. Frontiers in Sustainable Food Systems.
- IITA. Pest control in cassava farms (IPM field guide). https://www.iita.org/wp-content/uploads/2016/06/Pest_control_in_cassava_farms.pdf
