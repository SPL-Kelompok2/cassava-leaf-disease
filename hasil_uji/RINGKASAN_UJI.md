# Ringkasan uji (28 November 2026)

Semua angka di berkas ini berasal dari pengujian di mesin uji berbasis CPU. Model yang dipakai untuk menguji foto
**bukan** model VOLO-D2 pada laporan (akurasi 0,9035). Lihat bagian 4.

## 1. Lingkungan

| Komponen | Versi |
|---|---|
| Python | 3.13.18 |
| Java | 21 (untuk Pellet) |
| owlready2 / rdflib | 0.37 / 7.6.0 |
| torch / timm | 2.15.0 / 1.0.31 |
| Streamlit | 1.67.0 (dengan torch) dan 1.65.0 (tanpa torch) |
| pandas / numpy / pytest | 3.1.1 / 2.5.5 / 9.2.0 |

## 2. Tes otomatis (`pytest`)

| Versi kode | Dengan torch | Tanpa torch |
|---|---|---|
| Asli (dari zip) | 142 lulus | 114 lulus, 1 gagal, 5 dilewati |
| Sesudah perbaikan | **179 lulus** | **149 lulus**, 0 gagal, 6 dilewati |

- Ada 37 tes tambahan pada hitungan dengan torch.
- Satu-satunya kegagalan pada kode asli tanpa torch: `test_app_foto_checkpoint_tidak_ada_memberi_peringatan`.
- Keenam tes yang dilewati membutuhkan torch atau timm.
- Checksum T-Box termasuk dalam tes dan lulus.

## 3. Notebook Kaggle

`notebooks/train_vision_kaggle.ipynb` dijalankan sel demi sel, sekali dengan `VAL_SIZE = 0.0` dan sekali dengan
`VAL_SIZE = 0.1`. Semua sel kode selesai tanpa galat.

Penyesuaian untuk uji lokal: sel `pip install` dilewati, model diganti `resnet18` berukuran 64 piksel tanpa bobot
awal, 150 foto (30 per kelas), 2 epoch. Uji ini hanya memeriksa alur, sehingga akurasinya tidak bermakna. Pada mode
validasi, riwayat memuat `val_accuracy` dan `val_macro_f1`, dan `best_epoch` tercatat. Notebook ini **tidak**
dijalankan di Kaggle atau dengan GPU.

## 4. Uji dengan foto singkong nyata dari internet

### 4.1 Sumber foto

- iCassava 2019, salinan pada https://github.com/ai-agriculture-circuits-and-systems/cassava
  (commit `91944398c4f46730ff465fd8843746d4c79c6392`).
- 5.925 foto berlabel: folder `train` 5.656, `validation` 163, `test` 106.
- Lisensi perlu diperhatikan: README repositori menyebut MIT, berkas LICENSE menyebut CC BY 4.0 (© 2019 icassava).
- Foto asli tidak disertakan di sini, kecuali yang tampak pada tangkapan layar dan `contoh_duplikat.jpg`.

### 4.2 Pemeriksaan citra ganda (`scripts/check_duplicates.py`)

| Ukuran | Identik (SHA-256) | Identik atau mirip (dHash <= 4) |
|---|---|---|
| Grup ganda | 15 | 39 |
| Citra di dalam grup | 30 | 81 |
| Salinan berlebih | 15 | 42 |
| Grup berlabel bentrok | 13 | 17 |
| Grup menyeberang folder train/validation/test | 0 | 3 |

- Tiga grup yang menyeberang folder: `train-cbb-222` dengan `validation/cbb.7`; `test/cbsd.23` dengan
  `train-cbsd-213`; `train-cbsd-500` dengan `validation/cbsd.31`.
- `foto_nyata/contoh_duplikat.jpg` memperlihatkan empat pasangan yang sudah dicek mata, termasuk satu foto yang sama
  dengan label CBB dan CBSD. Grup "mirip" lainnya belum dicek satu per satu.
- Pemeriksaan ini memakai folder bawaan dataset. Pembagian 80/20 yang menghasilkan 0,9035 **belum diperiksa**.

### 4.3 Model uji: lapisan linear di atas VOLO-D2 384 beku

Tujuannya menguji jalur foto ke aplikasi dengan model yang benar-benar belajar dari foto singkong. Backbone memakai
bobot ImageNet resmi VOLO dan tidak dilatih ulang; hanya satu lapisan linear yang dilatih. Tidak ada augmentasi.

1. Duplikat dan citra mirip dibuang **sebelum** pembagian; grup berlabel bentrok dibuang seluruhnya
   (5.925 menjadi 5.866).
2. Grup yang menyentuh folder `train` dibagi 70/10/20 per kelas: latih 3.921, validasi 560, uji 1.119.
3. Sisanya, dari folder `validation` dan `test`, disebut `luar` (266 foto).
4. `weight_decay` dipilih dari macro F1 validasi (terpilih 0,01). Data uji dan `luar` tidak dipakai untuk memilih.

| Bagian | n | Akurasi | Macro F1 | CI 95% akurasi | Tebak kelas mayoritas |
|---|---|---|---|---|---|
| Latih | 3.921 | 0,8712 | 0,8162 | - | - |
| Validasi | 560 | 0,8250 | 0,7558 | - | - |
| **Uji** | 1.119 | **0,8239** | **0,7372** | 0,8007-0,8454 | 0,4727 |
| **Luar** | 266 | **0,8835** | **0,8624** | 0,8459-0,9211 | 0,2481 |

CI dihitung dengan bootstrap per citra (2.000 ulangan).

| Kelas | Uji: presisi | recall | F1 | n | Luar: presisi | recall | F1 | n |
|---|---|---|---|---|---|---|---|---|
| CBB | 0,603 | 0,385 | 0,470 | 91 | 0,870 | 0,571 | 0,690 | 35 |
| CBSD | 0,772 | 0,824 | 0,797 | 284 | 0,810 | 0,879 | 0,843 | 58 |
| CGM | 0,743 | 0,724 | 0,733 | 152 | 0,976 | 0,837 | 0,901 | 49 |
| CMD | 0,906 | 0,934 | 0,920 | 529 | 0,880 | 1,000 | 0,936 | 66 |
| Healthy | 0,754 | 0,778 | 0,766 | 63 | 0,905 | 0,983 | 0,942 | 58 |

Matriks konfusi data uji (baris = label, kolom = prediksi):

| | CBB | CBSD | CGM | CMD | Healthy |
|---|---|---|---|---|---|
| CBB | 35 | 36 | 9 | 7 | 4 |
| CBSD | 14 | 234 | 12 | 20 | 4 |
| CGM | 2 | 16 | 110 | 19 | 5 |
| CMD | 5 | 11 | 16 | 494 | 3 |
| Healthy | 2 | 6 | 1 | 5 | 49 |

Checkpoint dimuat lewat `VisionModel` milik aplikasi pada 266 foto `luar`: akurasi sama (0,8835), selisih
probabilitas terbesar 6,9e-07, sekitar 0,50 detik per foto.

**Batas penafsiran:**

- CBB lemah: pada data uji, 36 dari 91 foto CBB diprediksi CBSD.
- Data `luar` berasal dari dataset yang sama, jadi bukan uji lintas daerah atau lintas kamera. Komposisi kelasnya
  juga jauh lebih seimbang daripada data uji, sehingga kedua akurasi tidak dapat dibandingkan langsung.
- Foto berbeda dari tanaman yang sama masih mungkin berada di bagian yang berbeda.
- Angka ini tidak dapat dibandingkan dengan 0,9035 pada laporan atau 0,8964 pada paper.

### 4.4 Uji antarmuka di peramban (Chrome)

Aplikasi dijalankan dengan checkpoint uji di atas. Foto dipilih otomatis: foto pertama tiap kelas pada bagian `luar`
menurut urutan nama.

| Skenario | Hasil | Tangkapan layar |
|---|---|---|
| Foto `cbb.0` | Prediksi **CBSD, salah** (label CBB); skor 37%, ditandai kecocokan rendah | 01 |
| Foto `cbsd.0`, `cgm.0`, `cmd.0`, `healthy.0` | Prediksi sesuai label | 02-05 |
| Fusi foto CMD dan contoh data CMD | Kesimpulan CMD; teks Eq. (1) dan catatan R14 tampil | 06, 07 |
| Data lahan yang menghasilkan skor seri | "Belum konklusif" dengan CBB dan CGM | 08, 09 |
| Kedua bobot 0 | Pesan penolakan, tanpa crash | 10 |
| Vektor visi nol semua | Pesan penolakan | 11 |
| Mode pakar | Catatan R14 dan catatan CGM pada CQ5 tampil | 12, 13 |
| Tab Tentang | Keterangan asal angka kinerja tampil | 14 |

Kelima kesimpulan di layar sama dengan prediksi yang dihitung di luar aplikasi. Tidak ada kotak exception atau galat
halaman, dan log Streamlit tidak memuat galat. Log: `layar/log_uji_antarmuka.txt`.

Uji ini menemukan grafik *Rincian angka* yang menumpuk tiga skor. Grafik sudah diperbaiki, diberi tes, dan uji
peramban diulang.

## 5. Yang tidak diuji

- Model VOLO-D2 pada laporan (checkpoint dan dataset gabungannya tidak ada di zip).
- Pelatihan penuh dengan GPU.
- Fusi pada pasangan foto dan sensor yang nyata. Data lahan pada skenario fusi adalah contoh bawaan aplikasi.
- Foto yang bukan daun singkong.
- Kamera ponsel dan peramban selain Chrome.

## 6. Cara mengulang

Checkpoint uji (sekitar 231 MB) tidak disertakan. Untuk membuatnya kembali, unduh dataset di atas dan bobot
https://github.com/sail-sg/volo/releases/download/volo_1/d2_384_86.0.pth.tar
(SHA-256 `870c27721c5f7bc7d5650b4cacb50476fa2e26566a9a412aae218c590d942852`), lalu jalankan dari folder proyek (perintah untuk bash, misalnya di Linux atau Git Bash):

```bash
python hasil_uji/skrip/ekstrak_fitur.py . d2_384_86.0.pth.tar DIR_DATASET/data fitur
python hasil_uji/skrip/uji_foto_nyata.py --proyek . --data DIR_DATASET/data \
    --bobot d2_384_86.0.pth.tar --fitur fitur --keluar keluaran

python scripts/check_duplicates.py --csv hasil_uji/foto_nyata/semua.csv --img-dir DIR_DATASET/data \
    --split hasil_uji/foto_nyata/folder_split.csv --report laporan_duplikat.csv

# terminal 1
CASSAVA_VISION_CKPT=keluaran/vision_linear_probe.pt streamlit run app.py --server.port 8613
# terminal 2 (perlu paket playwright dan Google Chrome di /usr/bin/google-chrome)
mkdir -p layar
python hasil_uji/skrip/ui_uji.py http://localhost:8613 DIR_DATASET/data \
    keluaran/prediksi_luar_lewat_aplikasi.csv layar
```

Ekstraksi fitur berjalan di CPU dan memakan waktu lama.
