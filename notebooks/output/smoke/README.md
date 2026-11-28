# Folder `smoke`: uji cepat, BUKAN hasil kinerja

Isi folder ini berasal dari sel *uji cepat* notebook Kaggle (`--limit 300 --epochs 1`): 240 citra latih,
60 citra uji, satu epoch. Tujuannya hanya memastikan pipa pelatihan berjalan tanpa galat.

Model pada uji ini menebak semua citra uji sebagai CMD, sehingga akurasi 0,6167 hanya mencerminkan porsi
kelas CMD pada 60 citra tersebut. Jangan kutip angka ini sebagai akurasi model.

Hasil pelatihan referensi ada di folder `../vision/` (VOLO-D2 384, 10 epoch).
