"""Metadata of the harmonized HRU dataset: global attributes and, per variable, long name, units, source and references.

`add_metadata` sets these attributes on a dataset; preprocessing.py calls it before writing the dataset, and
add_dataset_metadata.py writes them to an existing dataset without rebuilding it.
"""

import warnings

import xarray as xr

REFERENCES = {
    "foen_2010": (
        "FOEN: Topographische Einzugsgebiete Schweizer Gewässer: Teileinzugsgebiete 2 km², "
        "http://www.bafu.admin.ch/ezgg-ch, last access: 25 September 2025, 2010."
    ),
    "hoge_2023": (
        "Höge, M., Kauzlaric, M., Siber, R., Schönenberger, U., Horton, P., Schwanbeck, J., Floriancic, M. G., "
        "Viviroli, D., Wilhelm, S., Sikorska-Senoner, A. E., Addor, N., Brunner, M., Pool, S., Zappa, M., and "
        "Fenicia, F.: CAMELS-CH: hydro-meteorological time series and landscape attributes for 331 catchments in "
        "hydrologic Switzerland, Earth System Science Data, 15, 5755–5784, https://doi.org/10.5194/essd-15-5755-2023, "
        "2023."
    ),
    "euhydro_2020": (
        "Copernicus Land Monitoring Service: EU-Hydro River Network Database 2006-2012 (vector), Europe - version 1.3, "
        "Nov. 2020, https://sdi.eea.europa.eu/catalogue/copernicus/api/records/393359a7-7ebd-4a52-80ac-1a18d5f3db9c, "
        "last access: 14 October 2025, 2020."
    ),
    "kauzlaric_2023": (
        "Kauzlaric, M., Schürmann, S., Ummel, D., and Zischg, A.: Hourly discharge database HydroCH, Zenodo, "
        "https://doi.org/10.5281/zenodo.7691294, 2023."
    ),
    "klingler_2021": (
        "Klingler, C., Schulz, K., and Herrnegger, M.: LamaH-CE: LArge-SaMple DAta for Hydrology and Environmental "
        "Sciences for Central Europe, Earth System Science Data, 13, 4529–4565, "
        "https://doi.org/10.5194/essd-13-4529-2021, 2021."
    ),
    "vorarlberg_2025": (
        "Land Vorarlberg: Wasserwirtschaft, https://vorarlberg.at/-/wasserwirtschaft-viid, last access: "
        "16 November 2025, 2025."
    ),
    "lfu_bayern_2025": (
        "LfU Bayern: Gewässerkundlicher Dienst Bayern, https://www.gkd.bayern.de/en/rivers/discharge/tables, last "
        "access: 16 November 2025, 2025."
    ),
    "lubw_2025": (
        "LUBW: Daten- und Kartendienst der LUBW, https://udo.lubw.baden-wuerttemberg.de/public/p/pegel_messwerte_leer, "
        "last access: 16 November 2025, 2025."
    ),
    "hydroportail_2025": (
        "French ministry of ecology: SC Vigicrues - HydroPortail, https://www.hydro.eaufrance.fr/, last access: "
        "16 November 2025, 2025."
    ),
    "arpa_lombardia_2025": (
        "ARPA Lombardia: Dati e Indicatori ARPA Lombardia, https://www.arpalombardia.it/dati-e-indicatori/, last "
        "access: 16 November 2025, 2025."
    ),
    "arpa_piemonte_2025": (
        "ARPA Piemonte: Geoportale Arpa Piemonte, https://geoportale.arpa.piemonte.it/app/public/, last access: "
        "16 November 2025, 2025."
    ),
    "viviroli_2009": (
        "Viviroli, D., Zappa, M., Gurtz, J., and Weingartner, R.: An introduction to the hydrological modelling system "
        "PREVAH and its pre- and post-processing-tools, Environmental Modelling & Software, 24, 1209–1222, "
        "https://doi.org/10.1016/j.envsoft.2009.04.001, 2009."
    ),
    "schwarb_2000": (
        "Schwarb, M.: The alpine precipitation climate: evaluation of a high-resolution analysis scheme using "
        "comprehensive rain-gauge data, Doctoral Thesis, ETH Zurich, https://doi.org/10.3929/ethz-a-004121274, 2000."
    ),
    "meteoswiss_rhiresd_2021": (
        "MeteoSwiss: Daily Precipitation (final analysis): RhiresD, "
        "https://www.meteoschweiz.admin.ch/dam/jcr:4f51f0f1-0fe3-48b5-9de0-15666327e63c/ProdDoc_RhiresD.pdf, last "
        "access: 1 April 2024, 2021."
    ),
    "frei_2014": (
        "Frei, C.: Interpolation of temperature in a mountainous region using nonlinear profiles and non-Euclidean "
        "distances, International Journal of Climatology, 34, 1585–1605, https://doi.org/10.1002/joc.3786, 2014."
    ),
    "meteoswiss_tabsd_2021": (
        "MeteoSwiss: Daily Mean, Minimum and Maximum Temperature: TabsD, TminD, TmaxD, "
        "https://www.meteoschweiz.admin.ch/dam/jcr:818a4d17-cb0c-4e8b-92c6-1a1bdf5348b7/ProdDoc_TabsD.pdf, last "
        "access: 1 April 2024, 2021."
    ),
    "era5land_2019": (
        "Copernicus Climate Change Service: ERA5-Land hourly data from 1950 to present, Copernicus Climate Change "
        "Service (C3S) Climate Data Store (CDS), https://doi.org/10.24381/CDS.E2161BAC, 2019."
    ),
    "poggio_2021": (
        "Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, B., Ribeiro, E., and Rossiter, D.: "
        "SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty, SOIL, 7, "
        "217–240, https://doi.org/10.5194/soil-7-217-2021, 2021."
    ),
    "corine_2020": (
        "Copernicus Land Monitoring Service: CORINE Land Cover 2018 (raster 100 m), Europe, 6-yearly, version "
        "2020_20u1, https://doi.org/10.2909/960998c1-1870-4e82-8051-6485205ebbac, last access: 29 September 2025, "
        "2020."
    ),
    "yamazaki_2019": (
        "Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., and Pavelsky, T. M.: MERIT Hydro: A "
        "High-Resolution Global Hydrography Map Based on Latest Topography Dataset, Water Resources Research, 55, "
        "5053–5073, https://doi.org/10.1029/2019WR024873, 2019."
    ),
    "horn_1981": (
        "Horn, B. K. P.: Hill shading and the reflectance map, Proceedings of the IEEE, 69, 14–47, "
        "https://doi.org/10.1109/PROC.1981.11918, 1981."
    ),
    "barnes_2016": "Barnes, R.: RichDEM: Terrain Analysis Software, 2016.",
    "sfoe_2023": (
        "SFOE: Stauanlagen unter Bundesaufsicht, "
        "https://www.geocat.ch/geonetwork/srv/ger/catalog.search#/metadata/b5225e07-9884-47b9-8bc2-161d07d1e7f2, last "
        "access: 29 September 2025, 2023."
    ),
    "zischg_2019": (
        "Zischg, A. P., Kauzlaric, M., Schick, S., Mosimann, M., and Schürmann, S.: MAARE+ Lakes and lake regulations "
        "(available upon request), Mobiliar Lab for Natural Risks, Oeschger Centre for Climate Change Research, "
        "University of Bern, Bern, Switzerland, 2019."
    ),
}


def references(*keys: str) -> str:
    """Join the references for `keys` into one numbered string, as used in the `references` attribute."""
    if len(keys) == 1:
        return REFERENCES[keys[0]]
    return " ".join(f"{i}) {REFERENCES[key]}" for i, key in enumerate(keys, start=1))


GLOBAL_ATTRS = {
    "title": "Harmonized HRU dataset for DROP",
    "author": "Basil Kraft",
    "institution": "ETH Zurich",
    "source": "Harmonized from third-party data; see the `source` and `references` attributes of each variable.",
    "comment": (
        "Data aggregated to hydrological drainage units (HRUs) covering hydrological Switzerland. Gridded data were "
        "aggregated by the spatial average within each HRU, the circular mean for aspect, and the nearest neighbour "
        "where no grid cell had its centre within an HRU."
    ),
    "references": (
        "Kraft, B., et al.: DROP: A scalable deep learning approach for runoff simulation and river routing, Water "
        "Resources Research, 2026."
    ),
    "license": (
        "Open Database License (ODbL) 1.0, https://opendatacommons.org/licenses/odbl/1-0/. Required because the "
        "dataset contains data derived from MERIT Hydro, which is licensed under ODbL 1.0 or CC BY-NC 4.0."
    ),
    "acknowledgement": (
        "Contains modified data from the sources listed in the `source` and `references` attributes of each variable. "
        "Source: MeteoSwiss. Source: FOEN. Source: SFOE. Contains modified Copernicus Climate Change Service "
        "information (ERA5-Land) and modified Copernicus Land Monitoring Service information (CORINE Land Cover, "
        "EU-Hydro), produced with funding by the European Union. Neither the European Union nor the data providers "
        "endorse this dataset."
    ),
    "history": "Created with project/scripts/preprocessing.py of https://github.com/bask0/drop-ch.",
}

DU = ("foen_2010", "hoge_2023")
DU_CONNECTIVITY = (*DU, "euhydro_2020")
DISCHARGE = (
    "kauzlaric_2023",
    "klingler_2021",
    "vorarlberg_2025",
    "lfu_bayern_2025",
    "lubw_2025",
    "hydroportail_2025",
    "arpa_lombardia_2025",
    "arpa_piemonte_2025",
)
PREVAH = ("viviroli_2009",)
TOPOGRAPHY = ("yamazaki_2019", "horn_1981", "barnes_2016")
DU_SOURCE = "FOEN; CAMELS-CH outside of Switzerland"
DISCHARGE_SOURCE = "HydroCH, compiled from national and regional providers"
PREVAH_SOURCE = "PREVAH hydrological model simulations"
DAM_SOURCE = "SFOE: Stauanlagen unter Bundesaufsicht"
SOIL_COMMENT = "Depth-weighted mean over 0 to 30 cm."
DAM_COMMENT = "0, empty or False for HRUs without a dam or reservoir."
DAM_UNITS_COMMENT = "In the units of the source dataset. " + DAM_COMMENT

VARIABLE_ATTRS = {
    # Drainage units and their connectivity.
    "hru": dict(long_name="HRU ID", source=DU_SOURCE, references=references(*DU)),
    "area": dict(
        long_name="HRU area",
        units="km2",
        source="Derived from the HRU polygons",
        references=references(*DU),
    ),
    "upstream_area": dict(
        long_name="Upstream area",
        units="km2",
        source="Derived from the HRU polygons and their connectivity",
        references=references(*DU_CONNECTIVITY),
        comment="Total area of the HRU and all HRUs upstream of it.",
    ),
    "downstream": dict(
        long_name="Downstream HRU",
        source=DU_SOURCE + "; connectivity outside of Switzerland based on EU-Hydro",
        references=references(*DU_CONNECTIVITY),
        comment="ID of the HRU that the HRU drains into, -1 if none.",
    ),
    "x": dict(
        long_name="x coordinate of the HRU centroid",
        units="m",
        source="Derived from the HRU polygons",
        references=references(*DU),
        comment="Swiss coordinate system CH1903+/LV95 (EPSG:2056).",
    ),
    "y": dict(
        long_name="y coordinate of the HRU centroid",
        units="m",
        source="Derived from the HRU polygons",
        references=references(*DU),
        comment="Swiss coordinate system CH1903+/LV95 (EPSG:2056).",
    ),
    "is_lake": dict(
        long_name="Lake indicator",
        source=DU_SOURCE,
        references=references(*DU),
        comment="True if the HRU is a standing water body.",
    ),
    # Discharge observations.
    "Q_obs_m3s": dict(
        long_name="Observed discharge",
        units="m3 s-1",
        source=DISCHARGE_SOURCE,
        references=references(*DISCHARGE),
        comment="Daily mean of hourly observations, assigned to the HRU at the gauge; missing elsewhere.",
    ),
    "Q_obs_mmd": dict(
        long_name="Observed discharge per upstream area",
        units="mm d-1",
        source=DISCHARGE_SOURCE,
        references=references(*DISCHARGE),
        comment="Q_obs_m3s divided by upstream_area.",
    ),
    "has_obs": dict(
        long_name="Discharge observations available",
        source=DISCHARGE_SOURCE,
        references=references(*DISCHARGE),
        comment="True if Q_obs_m3s has at least one value.",
    ),
    "Q_emulated": dict(
        long_name="Emulated discharge (deprecated)",
        comment="Not used by the current code; kept for compatibility with earlier code versions.",
    ),
    # PREVAH runoff simulations.
    "q_prevah_mmd": dict(
        long_name="Runoff (PREVAH)",
        units="mm d-1",
        source=PREVAH_SOURCE,
        references=references(*PREVAH),
    ),
    "q_prevah_m3s": dict(
        long_name="Runoff (PREVAH)",
        units="m3 s-1",
        source=PREVAH_SOURCE,
        references=references(*PREVAH),
        comment="q_prevah_mmd multiplied by area.",
    ),
    "has_prevah": dict(
        long_name="PREVAH coverage",
        source=PREVAH_SOURCE,
        references=references(*PREVAH),
        comment="True if PREVAH covers at least half of the HRU.",
    ),
    # Meteorological forcing.
    "prec": dict(
        long_name="Precipitation",
        units="mm d-1",
        source="MeteoSwiss: RhiresD",
        references=references("schwarb_2000", "meteoswiss_rhiresd_2021"),
    ),
    "tair": dict(
        long_name="Daily mean air temperature",
        units="degC",
        source="MeteoSwiss: TabsD; ERA5-Land outside of Switzerland",
        references=references("frei_2014", "meteoswiss_tabsd_2021", "era5land_2019"),
        comment=(
            "Outside of Switzerland, ERA5-Land air temperature mapped to TabsD with a neural network trained where "
            "both products overlap."
        ),
    ),
    # Soil properties.
    "bdod_mean": dict(
        long_name="Bulk density of the fine earth fraction",
        units="cg cm-3",
        source="SoilGrids 2.0",
        references=references("poggio_2021"),
        comment=SOIL_COMMENT,
    ),
    "cfvo_mean": dict(
        long_name="Volumetric fraction of coarse fragments",
        units="cm3 dm-3",
        source="SoilGrids 2.0",
        references=references("poggio_2021"),
        comment=SOIL_COMMENT,
    ),
    "clay_mean": dict(
        long_name="Clay content",
        units="g kg-1",
        source="SoilGrids 2.0",
        references=references("poggio_2021"),
        comment=SOIL_COMMENT,
    ),
    "sand_mean": dict(
        long_name="Sand content",
        units="g kg-1",
        source="SoilGrids 2.0",
        references=references("poggio_2021"),
        comment=SOIL_COMMENT,
    ),
    "silt_mean": dict(
        long_name="Silt content",
        units="g kg-1",
        source="SoilGrids 2.0",
        references=references("poggio_2021"),
        comment=SOIL_COMMENT,
    ),
    # Dams and reservoirs.
    "dam_name": dict(long_name="Dam name", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT),
    "dam_type_name": dict(
        long_name="Dam type", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "dam_height": dict(
        long_name="Dam height", units="m", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "construction_year": dict(
        long_name="Construction year", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "beginning_of_operation": dict(
        long_name="Beginning of operation", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "impoundment_volume": dict(
        long_name="Impoundment volume", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_UNITS_COMMENT
    ),
    "impoundment_level": dict(
        long_name="Impoundment level", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_UNITS_COMMENT
    ),
    "storage_level": dict(
        long_name="Storage level", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_UNITS_COMMENT
    ),
    "facility_aim_name": dict(
        long_name="Facility aim", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "is_storage_dam": dict(
        long_name="Storage dam", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "is_barrage": dict(
        long_name="Barrage",
        source=DAM_SOURCE,
        references=references("sfoe_2023"),
        comment="Managed lakes are regulated by barrages, so they are also flagged. " + DAM_COMMENT,
    ),
    "is_hydroelectricity": dict(
        long_name="Facility aim: hydropower", source=DAM_SOURCE, references=references("sfoe_2023"), comment=DAM_COMMENT
    ),
    "is_floodcontrol": dict(
        long_name="Facility aim: flood control",
        source=DAM_SOURCE,
        references=references("sfoe_2023"),
        comment=DAM_COMMENT,
    ),
    "is_other_aim": dict(
        long_name="Facility aim: other",
        source=DAM_SOURCE,
        references=references("sfoe_2023"),
        comment="Recreation and water supply. " + DAM_COMMENT,
    ),
    "is_regulated_lake": dict(
        long_name="Regulated lake",
        source="MAARE+",
        references=references("zischg_2019"),
    ),
}

# Topography from the MERIT DEM, reprojected to 50 m.
for stat in ["mean", "min", "max"]:
    VARIABLE_ATTRS[f"height_{stat}"] = dict(
        long_name=f"Elevation ({stat})",
        units="m",
        source="MERIT Hydro",
        references=references("yamazaki_2019"),
    )
    VARIABLE_ATTRS[f"slope_{stat}"] = dict(
        long_name=f"Slope ({stat})",
        units="1",
        source="Derived from MERIT Hydro with RichDEM",
        references=references(*TOPOGRAPHY),
        comment=(
            "Rise over run, computed without passing the 50 m cell size to RichDEM, which then assumes 1 m cells, so "
            "the values are 50 times the slope in m m-1."
        ),
    )
VARIABLE_ATTRS["aspect_mean"] = dict(
    long_name="Aspect (circular mean)",
    units="degree",
    source="Derived from MERIT Hydro with RichDEM",
    references=references(*TOPOGRAPHY),
)

# Land cover from CORINE, reclassified into eight classes.
for land_cover in ["artificial", "agriculture", "forests", "shrubs", "barren", "glaciers", "wetlands", "water"]:
    VARIABLE_ATTRS[f"{land_cover}_freq"] = dict(
        long_name=f"Areal fraction of {land_cover}",
        units="1",
        source="CORINE Land Cover 2018, reclassified",
        references=references("corine_2020"),
    )


# Columns of the HRU polygons (`hru_file`) that are not variables of the dataset.
FOEN_COLUMNS = [
    "TEILEZGNR",
    "H1",
    "H2",
    "Nebenarm",
    "InterneSen",
    "Gewaessern",
    "TEZGNR40",
    "TEZGNR150",
    "TEZGNR1000",
    "Gebietsaus",
    "SEE",
    "FLUSSGB",
]
HRU_COLUMN_DESCRIPTIONS = {
    "EZGNR": "HRU ID, as `hru` in the dataset. Source: " + DU_SOURCE + ".",
    "origin": "Origin of the polygon: 'ezg' for FOEN drainage units, 'camels' for CAMELS-CH catchments.",
    "Shape_Area": "Polygon area (m2).",
    **dict.fromkeys(FOEN_COLUMNS, "Attribute of the FOEN drainage units; see the FOEN product documentation."),
}


def hru_layer_metadata(columns: list[str]) -> dict[str, str]:
    """Return the layer metadata of the HRU polygons: global attributes and a description per column.

    The descriptions of columns that are also variables of the dataset are built from their attributes; the references
    of all columns are collected in `data_references`.
    """

    metadata = {
        **GLOBAL_ATTRS,
        "title": "HRU polygons of the harmonized HRU dataset for DROP",
        "comment": (
            "Polygons of the hydrological drainage units (HRUs) covering hydrological Switzerland, with their static "
            "attributes. FOEN drainage units within Switzerland, CAMELS-CH catchments outside of Switzerland."
        ),
    }

    reference_texts = [REFERENCES[key] for key in DU]
    for column in columns:
        if column in VARIABLE_ATTRS:
            attrs = VARIABLE_ATTRS[column]
            units = f" ({attrs['units']})" if "units" in attrs else ""
            source = f" Source: {attrs['source']}." if "source" in attrs else ""
            comment = f" {attrs['comment']}" if "comment" in attrs else ""
            metadata[f"description_{column}"] = f"{attrs['long_name']}{units}.{source}{comment}"
            reference_texts += [REFERENCES[key] for key in REFERENCES if REFERENCES[key] in attrs.get("references", "")]
        elif column in HRU_COLUMN_DESCRIPTIONS:
            metadata[f"description_{column}"] = HRU_COLUMN_DESCRIPTIONS[column]
        else:
            warnings.warn(f"No metadata defined for column {column}.", stacklevel=2)

    unique_references = list(dict.fromkeys(reference_texts))
    metadata["data_references"] = " ".join(f"{i}) {text}" for i, text in enumerate(unique_references, start=1))

    return metadata


def add_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Set the global attributes and the attributes of each known variable, and warn about variables without any."""
    ds.attrs.update(GLOBAL_ATTRS)

    missing = []
    for name in ds.variables:
        if name in VARIABLE_ATTRS:
            ds[name].attrs.update(VARIABLE_ATTRS[name])
        elif name != "time":
            missing.append(name)

    if missing:
        warnings.warn(f"No metadata defined for: {', '.join(map(str, missing))}.", stacklevel=2)

    return ds
