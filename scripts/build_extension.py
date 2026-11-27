"""Membangun cassava-extension.owl (ekstensi T-Box milik implementasi ini).

T-Box asli paper (cassava-disease.owl) TIDAK diubah. Ekstensi hanya menambah:
  - ObjectProperty  isFullyFavourableFor (Plant -> PlantDisease)
  - Class           HighRiskPlant == Plant and isFullyFavourableFor some PlantDisease
Dipakai untuk inferensi tingkat-2 (aturan berantai) dan klasifikasi DL oleh Pellet.
Jalankan:  python scripts/build_extension.py
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from owlready2 import World, ObjectProperty, locstr
from cassava_es.kb import SOSA_FILE, TBOX_FILE, EXT_FILE, EXT_IRI, _uri

world = World()
world.get_ontology(_uri(SOSA_FILE)).load()
tbox = world.get_ontology(_uri(TBOX_FILE)).load()
ext = world.get_ontology(EXT_IRI)
ext.imported_ontologies.append(tbox)
with ext:
    class isFullyFavourableFor(ObjectProperty):
        domain = [tbox.Plant]
        range = [tbox.PlantDisease]
    isFullyFavourableFor.comment = [locstr(
        "Ekstensi implementasi ini (bukan dari paper): tanaman memenuhi SEMUA kondisi "
        "yang mendukung suatu penyakit menurut aturan pakar; bukan berarti terinfeksi.", "id")]
    class HighRiskPlant(tbox.Plant):
        equivalent_to = [tbox.Plant & isFullyFavourableFor.some(tbox.PlantDisease)]
    HighRiskPlant.comment = [locstr(
        "Ekstensi implementasi ini: tanaman yang memenuhi semua kondisi pendukung minimal satu penyakit.", "id")]
ext.save(file=str(EXT_FILE), format="rdfxml")
print("tersimpan:", EXT_FILE)
