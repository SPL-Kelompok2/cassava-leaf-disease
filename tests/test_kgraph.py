import json

from cassava_es import kgraph


def _edges(kg):
    return {(kg.nodes[x["source"]]["label"], x["label"], kg.nodes[x["target"]]["label"]): x["kind"]
            for x in kg.links}


def test_tbox_memuat_hierarki_dan_relasi_paper():
    e = _edges(kgraph.tbox_graph())
    assert ("ViralPlantDisease", "subClassOf", "PlantDisease") in e
    assert ("CMDCondition", "subClassOf", "DiseaseCondition") in e
    assert ("Plant", "hasCondition", "DiseaseCondition") in e
    assert ("Plant", "isPlantedIn", "Soil") in e
    assert ("HighRiskPlant", "subClassOf", "Plant") in e  # ekstensi implementasi ini
    assert all(k == kgraph.SCHEMA for k in e.values())


def test_tbox_individu_opsional():
    kg = kgraph.tbox_graph()
    assert "CassavaMosaicDisease" not in kg.nodes
    kg = kgraph.tbox_graph(include_individuals=True)
    e = _edges(kg)
    assert ("CassavaMosaicDisease", "rdf:type", "ViralPlantDisease") in e
    assert ("CMDSoilMoisture", "enablesDisease", "CassavaMosaicDisease") in e
    assert kg.nodes["SoilPH"]["category"] == "Properti teramati"


def test_tbox_ringkasan():
    s = kgraph.tbox_summary()
    assert s["Object property"] >= 13 and s["Individu"] == 27


def test_abox_membedakan_fakta_masukan_dan_inferensi():
    kg = kgraph.abox_graph({"SoilMoisture": 0.6, "Temperature": 28.0}, True,
                           [("plant_1", "hasCondition", "CMDSoilMoisture"),
                            ("plant_1", "isFullyFavourableFor", "CassavaMosaicDisease")])
    e = _edges(kg)
    assert e[("plant_1", "isPlantedIn", "soil_1")] == kgraph.ASSERTED
    assert e[("obs_soil_1_SoilMoisture", "hasFeatureOfInterest", "soil_1")] == kgraph.ASSERTED
    assert e[("obs_field_1_Temperature", "hasFeatureOfInterest", "field_1")] == kgraph.ASSERTED
    assert e[("obs_soil_1_SoilMoisture", "hasSimpleResult", "0.6")] == kgraph.ASSERTED
    assert e[("field_1", "hasRainySeason", "true")] == kgraph.ASSERTED
    assert e[("plant_1", "hasCondition", "CMDSoilMoisture")] == kgraph.INFERRED
    assert e[("plant_1", "isFullyFavourableFor", "CassavaMosaicDisease")] == kgraph.INFERRED
    assert e[("CMDSoilMoisture", "enablesDisease", "CassavaMosaicDisease")] == kgraph.SCHEMA


def test_echarts_spec_valid_dan_bisa_diserialisasi():
    kg = kgraph.tbox_graph(include_individuals=True)
    spec = kgraph.to_echarts(kg, kgraph.TBOX_CATEGORIES)
    json.dumps(spec)
    series = spec["series"][0]
    ids = {n["id"] for n in series["data"]}
    assert len(ids) == len(series["data"])
    assert all(link["source"] in ids and link["target"] in ids for link in series["links"])
    assert all(0 <= n["category"] < len(series["categories"]) for n in series["data"])
