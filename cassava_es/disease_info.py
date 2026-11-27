"""Info umum tiap kelas hasil prediksi (definisi, gejala, penyebaran, pengendalian).

Teks ditulis untuk pengguna awam dan ditampilkan bersama setiap hasil prediksi; sumber
tercantum pada kolom ``sources`` tiap entri. Info disimpan di modul ini (bukan di T-Box)
agar ontologi tidak berubah.

Prinsip: informasi umum, bukan diagnosis atau resep (tanpa merek/dosis pestisida); angka
hanya dipakai bila ada di sumber; untuk konfirmasi di lapangan pengguna diarahkan ke
penyuluh pertanian atau dinas pertanian setempat.

Kelas "CGM": pada dataset iCassava (Mwebaze et al., 2019) dan literatur IITA artinya
*Cassava Green Mite*, tungau hama (*Mononychellus tanajoa*), sehingga info CGM mengikuti
makna tersebut; ontologi sistem ini menamainya "Cassava Green Mottle".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import CLASSES

GENERAL_DISCLAIMER = (
    "Informasi umum untuk edukasi, bukan diagnosis resmi. Prediksi berbasis foto dan data "
    "sensor dapat keliru (gejala beberapa penyakit mirip dan sering muncul bersamaan pada "
    "satu tanaman). Untuk memastikan penyakit dan menentukan penanganan, hubungi penyuluh "
    "pertanian atau dinas pertanian setempat."
)


@dataclass(frozen=True)
class DiseaseInfo:
    code: str
    name_en: str
    name_id: str
    category: str  # mis. "Penyakit virus", "Penyakit bakteri", "Hama (tungau)", "Sehat"
    agent: str  # penyebab
    definition: str  # apa itu (2-3 kalimat)
    symptoms: tuple  # ciri yang tampak
    spread: tuple  # cara menyebar
    favourable: str  # kondisi yang disebut mendukung (kualitatif)
    impact: str  # dampak
    management: tuple  # pengendalian umum
    notes: tuple = ()  # catatan/peringatan khusus kelas ini
    sources: tuple = field(default_factory=tuple)  # (judul, url)


_SRC_ICASSAVA = (
    "Mwebaze et al. (2019), iCassava 2019 Fine-Grained Visual Categorization Challenge",
    "https://arxiv.org/abs/1908.02900",
)
_SRC_CBB_REVIEW = (
    "Zárate-Chaves et al. (2021), Cassava diseases caused by Xanthomonas phaseoli pv. manihotis "
    "and X. cassavae, Molecular Plant Pathology",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8578842",
)
_SRC_CBB_UF = (
    "UF/IFAS EDIS, Cassava bacterial blight (lembar informasi)",
    "https://edis.ifas.ufl.edu/publication/VH053/pdf",
)
_SRC_VIRAL_REVIEW = (
    "Chikoti & Tembo (2022), Expansion and impact of cassava brown streak and cassava mosaic "
    "diseases in Africa: a review, Frontiers in Sustainable Food Systems",
    "https://doi.org/10.3389/fsufs.2022.1076364",
)
_SRC_IITA = (
    "IITA, Pest control in cassava farms (IPM field guide)",
    "https://www.iita.org/wp-content/uploads/2016/06/Pest_control_in_cassava_farms.pdf",
)
_SRC_INFONET = (
    "Infonet-Biovision, Cassava green spider mite",
    "https://www.infonet-biovision.org/PlantHealth/MinorPests/Cassava-green-spider-mite",
)

DISEASE_INFO: dict[str, DiseaseInfo] = {
    "CMD": DiseaseInfo(
        code="CMD",
        name_en="Cassava Mosaic Disease",
        name_id="Penyakit mosaik singkong",
        category="Penyakit virus",
        agent="Kelompok virus mosaik singkong (cassava mosaic begomovirus, famili Geminiviridae).",
        definition=(
            "Penyakit virus pada singkong yang membuat daun berbercak mosaik hijau-kuning dan "
            "pertumbuhan tanaman terhambat. Menurut makalah dataset iCassava, ini penyakit "
            "singkong yang paling tersebar luas di Afrika sub-Sahara."
        ),
        symptoms=(
            "Pola mosaik pada daun: bercak hijau normal bercampur kuning hingga putih "
            "(klorosis); proporsinya bergantung tingkat keparahan.",
            "Daun mengecil, berubah bentuk, dan helaiannya memuntir.",
            "Tanaman secara keseluruhan lebih kerdil.",
        ),
        spread=(
            "Ditularkan oleh kutu kebul Bemisia tabaci.",
            "Terbawa dan tersebar jauh lewat stek dari tanaman yang sudah terinfeksi.",
        ),
        favourable=(
            "Populasi kutu kebul yang tinggi mendorong penularan. Aturan sensor pada sistem ini "
            "mengaitkannya dengan suhu udara tinggi, kelembapan tanah cukup, suhu tanah sekitar "
            "20-32 °C, dan kelembapan udara tinggi (ambang dari Tabel 1 paper)."
        ),
        impact=(
            "Klorosis mengurangi klorofil sehingga fotosintesis dan hasil umbi menurun. Bersama "
            "CBSD, ini termasuk penyakit paling merusak pada singkong di Afrika Timur dan Tengah."
        ),
        management=(
            "Tanam stek bebas virus dari sumber tepercaya; hindari stek dari tanaman bergejala.",
            "Gunakan varietas yang tahan atau toleran bila tersedia.",
            "Pantau kebun secara rutin dan singkirkan tanaman bergejala sejak dini.",
            "Kendalikan populasi kutu kebul sesuai anjuran penyuluh setempat.",
        ),
        notes=(
            "Gejala bisa tertukar dengan serangan tungau hijau (CGM), yang juga menimbulkan "
            "bercak klorosis (lihat entri CGM).",
        ),
        sources=(_SRC_ICASSAVA, _SRC_VIRAL_REVIEW, _SRC_IITA),
    ),
    "CBSD": DiseaseInfo(
        code="CBSD",
        name_en="Cassava Brown Streak Disease",
        name_id="Penyakit garis coklat singkong",
        category="Penyakit virus",
        agent=(
            "Dua spesies virus: Cassava brown streak virus (CBSV) dan Ugandan cassava brown "
            "streak virus (UCBSV), genus Ipomovirus, famili Potyviridae."
        ),
        definition=(
            "Penyakit virus yang menimbulkan pita kuning atau nekrotik di sepanjang tulang daun "
            "dan dapat merusak umbi dari dalam. Pada makalah dataset iCassava, penyakit ini "
            "disebut yang paling parah saat itu."
        ),
        symptoms=(
            "Daun: pita kuning atau nekrotik mengikuti tulang daun yang dapat melebar dan menyatu "
            "menjadi bercak kuning besar.",
            "Umbi: area nekrotik coklat tua di dalam umbi dan ukuran umbi mengecil.",
            "Gejala pada daun/batang dapat muncul tanpa gejala pada umbi (menurut makalah "
            "dataset), sehingga umbi perlu dicek terpisah.",
        ),
        spread=(
            "Ditularkan oleh kutu kebul Bemisia tabaci.",
            "Tersebar lewat stek dari tanaman yang terinfeksi.",
            "Penularan oleh kutu kebul tergolong kurang efisien dan bergantung pada populasi "
            "kutu kebul yang tinggi.",
        ),
        favourable=(
            "Populasi kutu kebul yang tinggi. Aturan sensor pada sistem ini mengaitkannya dengan "
            "kelembapan udara sekitar 0,7-0,85, kelembapan tanah >= 0,1, dan suhu tanah "
            "10-32 °C (ambang dari Tabel 1 paper)."
        ),
        impact=(
            "Umbi yang bernekrosis menurunkan hasil dan kualitas panen; umbi bisa tidak layak "
            "konsumsi."
        ),
        management=(
            "Tanam stek bebas virus dari sumber tepercaya dan hindari stek dari tanaman bergejala.",
            "Varietas yang tersedia umumnya baru toleran (gejala lebih ringan), belum kebal; "
            "tanaman toleran masih bisa menjadi sumber penularan.",
            "Pantau kebun dan singkirkan tanaman bergejala; periksa umbi saat panen.",
            "Kendalikan kutu kebul sesuai anjuran penyuluh setempat.",
        ),
        notes=(
            "Penyakit ini dilaporkan terutama di Afrika sub-Sahara, dan model gambar dilatih pada "
            "foto dari Uganda.",
        ),
        sources=(_SRC_ICASSAVA, _SRC_VIRAL_REVIEW),
    ),
    "CBB": DiseaseInfo(
        code="CBB",
        name_en="Cassava Bacterial Blight",
        name_id="Hawar bakteri singkong",
        category="Penyakit bakteri",
        agent="Bakteri Xanthomonas phaseoli pv. manihotis (sebelumnya X. axonopodis pv. manihotis).",
        definition=(
            "Penyakit bakteri sistemik pada singkong yang menimbulkan bercak bersudut pada daun, "
            "layu, hingga mati pucuk. Penyakit ini umum di daerah lembap."
        ),
        symptoms=(
            "Bercak daun bersudut, basah/tembus cahaya (terlihat jelas dari sisi bawah daun), "
            "kemudian mencoklat dan kadang dikelilingi halo kuning.",
            "Daun muda layu, mengering, dan gugur lebih awal pada serangan berat.",
            "Getah/eksudat lengket kekuningan pada daun dan pucuk muda; pada batang dapat terlihat "
            "lesi dan pembuluh berwarna coklat.",
        ),
        spread=(
            "Terbawa stek dari tanaman terinfeksi (jalur utama masuk ke kebun baru).",
            "Alat potong yang terkontaminasi, termasuk golok untuk memanen dan memotong stek.",
            "Percikan air hujan di dalam kebun, serta sentuhan pada daun yang basah.",
        ),
        favourable=(
            "Cuaca basah dan lembap; penyebaran cepat pada musim hujan karena percikan hujan. "
            "Aturan sensor pada sistem ini mengaitkannya dengan musim hujan, kelembapan tanah "
            ">= 0,3, pH tanah 6,5-7,2, dan suhu tanah 25-30 °C (ambang dari Tabel 1 paper)."
        ),
        impact=(
            "Di berbagai literatur disebut sebagai salah satu penyakit singkong dengan kerugian "
            "terbesar; keparahannya bergantung varietas, kesuburan tanah, iklim, dan jumlah "
            "inokulum."
        ),
        management=(
            "Gunakan stek sehat dari tanaman yang tidak bergejala.",
            "Bersihkan/desinfeksi alat potong dan hindari bekerja di antara tanaman saat daun basah.",
            "Singkirkan dan musnahkan sisa tanaman terinfeksi; pertimbangkan pergiliran tanaman.",
            "Gunakan varietas yang lebih tahan bila tersedia.",
        ),
        sources=(_SRC_ICASSAVA, _SRC_CBB_REVIEW, _SRC_CBB_UF),
    ),
    "CGM": DiseaseInfo(
        code="CGM",
        name_en="Cassava Green Mite",
        name_id="Tungau hijau singkong",
        category="Hama (tungau), bukan penyakit patogen",
        agent="Tungau Mononychellus tanajoa (Tetranychidae), hama yang mengisap cairan daun dan pucuk.",
        definition=(
            "Kerusakan daun akibat tungau hijau yang hidup di sisi bawah daun muda. Pada dataset "
            "iCassava, kelas 'CGM' berarti Cassava Green Mite, yaitu hama, bukan virus."
        ),
        symptoms=(
            "Bintik klorosis kecil seukuran tusukan jarum pada permukaan atas daun, yang dapat "
            "membesar sampai seluruh daun memucat/putih.",
            "Daun muda menjadi kecil, sempit, dan cacat; daun yang rusak berat mengerut dan "
            "mengering.",
            "Pada serangan berat pucuk gundul dan tampak seperti 'tongkat lilin' (candle-stick).",
        ),
        spread=(
            "Tungau sangat kecil (tampak seperti bintik halus; lebih jelas dengan kaca pembesar) "
            "dan berkembang biak di sisi bawah daun muda serta pucuk.",
        ),
        favourable=(
            "Sumber agronomi (IITA) menyebut kerusakannya lebih parah di musim kering daripada "
            "musim hujan. Aturan sensor pada sistem ini (ambang Tabel 1) mengaitkan CGM dengan "
            "kelembapan tanah dan udara tinggi (>= 0,7) serta suhu tanah >= 27 °C."
        ),
        impact=(
            "Serangan berat dapat menurunkan hasil umbi secara nyata; rentang yang dilaporkan "
            "di literatur sekitar 13-80 %, tergantung kondisi."
        ),
        management=(
            "Gunakan varietas yang lebih toleran bila tersedia dan stek yang bersih.",
            "Tanam pada awal musim hujan agar tanaman tumbuh kuat; tanaman muda (sekitar 2-9 bulan) "
            "paling rentan.",
            "Musuh alami (tungau predator, mis. Typhlodromalus aripo) menekan populasi di Afrika; "
            "hindari tindakan yang merusak musuh alami dan konsultasikan opsi pengendalian dengan "
            "penyuluh.",
        ),
        notes=(
            "Kelas ini ditulis 'Cassava Green Mottle' pada ontologi sistem ini; pada dataset iCassava "
            "dan literatur IITA artinya Cassava Green Mite (tungau).",
            "Gejala CGM dapat tertukar dengan mosaik (CMD): bintik klorosis tungau tidak sama dengan "
            "bercak mosaik virus.",
        ),
        sources=(_SRC_ICASSAVA, _SRC_IITA, _SRC_INFONET),
    ),
    "Healthy": DiseaseInfo(
        code="Healthy",
        name_en="Healthy",
        name_id="Sehat",
        category="Sehat",
        agent="Tidak ada penyakit atau hama utama yang terdeteksi oleh model.",
        definition=(
            "Tidak ada ciri khas CMD, CBSD, CBB, atau CGM yang dominan pada data ini."
        ),
        symptoms=(
            "Daun berwarna hijau merata tanpa mosaik, pita kuning di tulang daun, bercak bersudut, "
            "atau bintik klorosis halus.",
        ),
        spread=(),
        favourable="-",
        impact="-",
        management=(
            "Tetap pantau kebun secara berkala; gejala awal bisa belum terlihat jelas.",
            "Gunakan stek bersih dan jaga sanitasi alat untuk mencegah masuknya penyakit.",
        ),
        notes=(
            "Hasil 'sehat' bukan jaminan bebas penyakit: foto daun tidak menampakkan infeksi yang "
            "belum bergejala atau gejala di bagian lain (batang, umbi).",
        ),
        sources=(_SRC_ICASSAVA,),
    ),
}

assert set(DISEASE_INFO) == set(CLASSES), "DISEASE_INFO harus mencakup semua kelas"


def get_info(code: str) -> DiseaseInfo:
    """Info untuk satu kelas (CBB/CBSD/CGM/CMD/Healthy)."""
    try:
        return DISEASE_INFO[code]
    except KeyError as exc:
        raise KeyError(f"kelas tidak dikenal: {code!r}") from exc
