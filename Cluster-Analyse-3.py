import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN


# ============================================================
# AT1 SPATIAL ANALYSIS
# QuPath CSV -> 6 AT1 spatial parameters
# ============================================================

st.set_page_config(
    page_title="AT1 Spatial Analysis",
    page_icon="?",
    layout="wide"
)

st.title("? AT1 Spatial Analysis")

st.markdown(
    """
    **QuPath CSV → AT1 spatial analysis**

    Die Analyse berechnet für jedes Bild:

    1. AT1-positive Zellen / mm²
    2. Clustered AT1 (%)
    3. Cluster / mm²
    4. Median AT1 / Cluster
    5. Median Clusterfläche (µm²)
    6. Voronoi CV
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Analyseparameter")

# ------------------------------------------------------------
# Pixelgröße
# ------------------------------------------------------------

pixel_um = st.sidebar.number_input(
    "Pixelgröße (µm / Pixel)",
    min_value=0.001,
    max_value=10.0,
    value=0.327,
    step=0.001,
    format="%.3f"
)

st.sidebar.markdown("---")

# ------------------------------------------------------------
# DBSCAN
# ------------------------------------------------------------

st.sidebar.subheader("Clusterdefinition")

eps_um = st.sidebar.number_input(
    "Clusterabstand eps (µm)",
    min_value=1.0,
    max_value=1000.0,
    value=50.0,
    step=5.0,
    help="Maximaler Abstand zwischen AT1-Zellen innerhalb eines Clusters."
)

min_samples = st.sidebar.number_input(
    "Minimale AT1-Zellen pro Cluster",
    min_value=2,
    max_value=50,
    value=3,
    step=1
)

st.sidebar.markdown("---")

# ------------------------------------------------------------
# Voronoi
# ------------------------------------------------------------

st.sidebar.subheader("Voronoi")

voronoi_percentile = st.sidebar.slider(
    "High-Density Schwelle (%)",
    min_value=1,
    max_value=50,
    value=20,
    step=1,
    help=(
        "Nur für zusätzliche Voronoi-Dichteinformation. "
        "Der Voronoi CV verwendet alle gültigen Voronoi-Flächen."
    )
)


# ============================================================
# CSV UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "QuPath CSV hochladen",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        """
        Bitte die von QuPath exportierte CSV-Datei hochladen.

        Erwartete Spalten:

        • Image
        • ROI_ID
        • ROI_Area
        • Positive_Count
        • X
        • Y
        """
    )

    st.stop()


# ============================================================
# CSV EINLESEN
# ============================================================

try:

    df = pd.read_csv(
        uploaded_file,
        sep=None,
        engine="python"
    )

except Exception as e:

    st.error(
        f"CSV konnte nicht gelesen werden:\n{e}"
    )

    st.stop()


# ============================================================
# SPALTEN PRÜFEN
# ============================================================

required_columns = [
    "Image",
    "ROI_ID",
    "ROI_Area",
    "X",
    "Y"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    st.error(
        "Folgende Spalten fehlen:\n\n"
        + ", ".join(missing)
    )

    st.write(
        "Gefundene Spalten:"
    )

    st.write(
        list(df.columns)
    )

    st.stop()


# ============================================================
# NUMERISCHE SPALTEN
# ============================================================

for col in [
    "ROI_Area",
    "X",
    "Y"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "Image",
        "ROI_ID",
        "ROI_Area",
        "X",
        "Y"
    ]
).copy()


# ============================================================
# HELPER: POLYGONFLÄCHE
# ============================================================

def polygon_area(vertices):

    vertices = np.asarray(
        vertices
    )

    x = vertices[:, 0]
    y = vertices[:, 1]

    return 0.5 * abs(
        np.dot(
            x,
            np.roll(y, 1)
        )
        -
        np.dot(
            y,
            np.roll(x, 1)
        )
    )


# ============================================================
# EIN BILD ANALYSIEREN
# ============================================================

def analyze_image(
    image_df,
    pixel_um,
    eps_um,
    min_samples
):

    image_name = str(
        image_df["Image"].iloc[0]
    )

    roi_id = str(
        image_df["ROI_ID"].iloc[0]
    )

    # --------------------------------------------------------
    # Zellzahl
    # --------------------------------------------------------

    n_cells = len(
        image_df
    )

    # --------------------------------------------------------
    # ROI-Fläche
    # --------------------------------------------------------

    roi_area_pixel2 = float(
        image_df["ROI_Area"].iloc[0]
    )

    roi_area_um2 = (
        roi_area_pixel2
        *
        pixel_um**2
    )

    roi_area_mm2 = (
        roi_area_um2
        /
        1_000_000
    )

    # --------------------------------------------------------
    # AT1 / mm²
    # --------------------------------------------------------

    if roi_area_mm2 > 0:

        at1_per_mm2 = (
            n_cells
            /
            roi_area_mm2
        )

    else:

        at1_per_mm2 = np.nan


    # ========================================================
    # ZU WENIGE ZELLEN
    # ========================================================

    if n_cells < 3:

        return {

            "Image": image_name,

            "ROI_ID": roi_id,

            "ROI_Area_mm2":
                roi_area_mm2,

            "AT1_per_mm2":
                at1_per_mm2,

            "Clustered_AT1_percent":
                0,

            "Clusters_per_mm2":
                0,

            "Median_AT1_per_Cluster":
                np.nan,

            "Median_Cluster_Area_um2":
                np.nan,

            "Voronoi_CV":
                np.nan,

            "AT1_Count":
                n_cells,

            "Cluster_Count":
                0
        }


    # ========================================================
    # KOORDINATEN
    # ========================================================

    xy_pixel = image_df[
        ["X", "Y"]
    ].to_numpy()

    xy_um = (
        xy_pixel
        *
        pixel_um
    )


    # ========================================================
    # DBSCAN
    # ========================================================

    eps_pixel = (
        eps_um
        /
        pixel_um
    )

    dbscan = DBSCAN(
        eps=eps_pixel,
        min_samples=int(
            min_samples
        )
    )

    labels = dbscan.fit_predict(
        xy_pixel
    )


    # ========================================================
    # CLUSTER
    # ========================================================

    cluster_ids = sorted(
        [
            x
            for x in np.unique(labels)
            if x != -1
        ]
    )

    cluster_count = len(
        cluster_ids
    )


    # --------------------------------------------------------
    # AT1 in Clustern
    # --------------------------------------------------------

    clustered_mask = (
        labels != -1
    )

    clustered_at1 = int(
        clustered_mask.sum()
    )

    if n_cells > 0:

        clustered_percent = (
            clustered_at1
            /
            n_cells
            *
            100
        )

    else:

        clustered_percent = np.nan


    # --------------------------------------------------------
    # Cluster / mm²
    # --------------------------------------------------------

    if roi_area_mm2 > 0:

        clusters_per_mm2 = (
            cluster_count
            /
            roi_area_mm2
        )

    else:

        clusters_per_mm2 = np.nan


    # ========================================================
    # CLUSTERGRÖSSEN
    # ========================================================

    cluster_sizes = []

    cluster_areas = []


    for cluster_id in cluster_ids:

        mask = (
            labels
            ==
            cluster_id
        )

        cluster_size = int(
            mask.sum()
        )

        cluster_sizes.append(
            cluster_size
        )


        # ----------------------------------------------------
        # Clusterfläche
        #
        # Wir verwenden die Fläche der konvexen Hülle
        # der Zellzentren.
        # ----------------------------------------------------

        cluster_points = xy_um[
            mask
        ]

        if cluster_size >= 3:

            try:

                from scipy.spatial import ConvexHull

                hull = ConvexHull(
                    cluster_points
                )

                area = float(
                    hull.volume
                )

            except Exception:

                area = np.nan

        else:

            # Bei 2 Punkten keine echte Fläche.
            # Wir setzen NaN.
            area = np.nan

        cluster_areas.append(
            area
        )


    # ========================================================
    # MEDIAN CLUSTERGRÖSSE
    # ========================================================

    if len(cluster_sizes) > 0:

        median_cluster_size = float(
            np.median(
                cluster_sizes
            )
        )

    else:

        median_cluster_size = np.nan


    # ========================================================
    # MEDIAN CLUSTERFLÄCHE
    # ========================================================

    valid_cluster_areas = [
        x
        for x in cluster_areas
        if not np.isnan(x)
        and x > 0
    ]

    if len(
        valid_cluster_areas
    ) > 0:

        median_cluster_area = float(
            np.median(
                valid_cluster_areas
            )
        )

    else:

        median_cluster_area = np.nan


    # ========================================================
    # VORONOI
    # ========================================================

    try:

        vor = Voronoi(
            xy_pixel
        )

        voronoi_areas = []

        for region_index in (
            vor.point_region
        ):

            region = vor.regions[
                region_index
            ]

            # ------------------------------------------------
            # Randzellen
            # ------------------------------------------------

            if (
                len(region) == 0
                or -1 in region
            ):

                voronoi_areas.append(
                    np.nan
                )

                continue


            vertices = vor.vertices[
                region
            ]

            area_pixel2 = polygon_area(
                vertices
            )

            area_um2 = (
                area_pixel2
                *
                pixel_um**2
            )

            voronoi_areas.append(
                area_um2
            )


        voronoi_areas = np.array(
            voronoi_areas
        )

        valid_voronoi = (
            ~np.isnan(
                voronoi_areas
            )
            &
            (
                voronoi_areas
                >
                0
            )
        )

        valid_areas = (
            voronoi_areas[
                valid_voronoi
            ]
        )


        # ----------------------------------------------------
        # Voronoi CV
        # ----------------------------------------------------

        if len(
            valid_areas
        ) >= 3:

            mean_area = np.mean(
                valid_areas
            )

            sd_area = np.std(
                valid_areas,
                ddof=1
            )

            if mean_area > 0:

                voronoi_cv = (
                    sd_area
                    /
                    mean_area
                )

            else:

                voronoi_cv = np.nan

        else:

            voronoi_cv = np.nan


    except Exception:

        voronoi_cv = np.nan


    # ========================================================
    # ERGEBNIS
    # ========================================================

    return {

        "Image":
            image_name,

        "ROI_ID":
            roi_id,

        "ROI_Area_mm2":
            roi_area_mm2,

        "AT1_Count":
            n_cells,

        "AT1_per_mm2":
            at1_per_mm2,

        "Clustered_AT1_percent":
            clustered_percent,

        "Clusters_per_mm2":
            clusters_per_mm2,

        "Median_AT1_per_Cluster":
            median_cluster_size,

        "Median_Cluster_Area_um2":
            median_cluster_area,

        "Voronoi_CV":
            voronoi_cv,

        "Cluster_Count":
            cluster_count
    }


# ============================================================
# ALLE BILDER ANALYSIEREN
# ============================================================

results = []


grouped = df.groupby(
    ["Image", "ROI_ID"],
    sort=True
)


progress = st.progress(
    0
)

total = len(
    grouped
)

for i, (
    group_key,
    image_df
) in enumerate(
    grouped
):

    result = analyze_image(
        image_df,
        pixel_um,
        eps_um,
        min_samples
    )

    results.append(
        result
    )

    progress.progress(
        int(
            (i + 1)
            /
            total
            *
            100
        )
    )


progress.empty()


# ============================================================
# ERGEBNISTABELLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SPALTEN SORTIEREN
# ============================================================

desired_columns = [

    "Image",

    "ROI_ID",

    "ROI_Area_mm2",

    "AT1_Count",

    "AT1_per_mm2",

    "Clustered_AT1_percent",

    "Clusters_per_mm2",

    "Median_AT1_per_Cluster",

    "Median_Cluster_Area_um2",

    "Voronoi_CV",

    "Cluster_Count"
]

results_df = results_df[
    desired_columns
]


# ============================================================
# ÜBERSICHT
# ============================================================

st.success(
    f"{len(results_df)} Bilder analysiert."
)


st.subheader(
    "? AT1-Ergebnisse pro Bild"
)


# ------------------------------------------------------------
# Rundung nur für Anzeige
# ------------------------------------------------------------

display_df = results_df.copy()

for col in [
    "ROI_Area_mm2",
    "AT1_per_mm2",
    "Clustered_AT1_percent",
    "Clusters_per_mm2",
    "Median_AT1_per_Cluster",
    "Median_Cluster_Area_um2",
    "Voronoi_CV"
]:

    display_df[col] = display_df[
        col
    ].round(3)


st.dataframe(
    display_df,
    use_container_width=True,
    height=600
)


# ============================================================
# KENNZAHLEN
# ============================================================

st.subheader(
    "? Übersicht"
)


c1, c2, c3, c4, c5, c6 = st.columns(
    6
)


c1.metric(
    "Bilder",
    len(results_df)
)

c2.metric(
    "AT1 gesamt",
    int(
        results_df[
            "AT1_Count"
        ].sum()
    )
)

c3.metric(
    "Ø AT1/mm²",
    f"{results_df['AT1_per_mm2'].mean():.1f}"
)

c4.metric(
    "Ø Clustered AT1",
    f"{results_df['Clustered_AT1_percent'].mean():.1f}%"
)

c5.metric(
    "Ø Cluster/mm²",
    f"{results_df['Clusters_per_mm2'].mean():.2f}"
)

c6.metric(
    "Ø Voronoi CV",
    f"{results_df['Voronoi_CV'].mean():.2f}"
)


# ============================================================
# DOWNLOAD
# ============================================================

st.subheader(
    "⬇️ Ergebnisse exportieren"
)


csv_output = results_df.to_csv(
    index=False
).encode(
    "utf-8-sig"
)


st.download_button(
    label="⬇️ AT1-Ergebnistabelle als CSV",
    data=csv_output,
    file_name="AT1_spatial_results.csv",
    mime="text/csv",
    width="stretch"
)


# ============================================================
# PARAMETER-ERKLÄRUNG
# ============================================================

st.markdown("---")

st.subheader(
    "? Parameter"
)

parameter_text = """

**AT1/mm²**

Anzahl AT1-positiver Zellen pro mm² analysierter ROI-Fläche.

**Clustered AT1 (%)**

Prozentualer Anteil der AT1-Zellen, die von DBSCAN einem Cluster zugeordnet werden.

**Cluster/mm²**

Anzahl der erkannten AT1-Cluster pro mm².

**Median AT1/Cluster**

Median der Anzahl AT1-Zellen innerhalb eines Clusters.

**Median Clusterfläche (µm²)**

Median der räumlichen Fläche der Cluster. Die Clusterfläche wird aus der konvexen Hülle der AT1-Zellzentren berechnet.

**Voronoi CV**

Variationskoeffizient der Voronoi-Flächen. Höhere Werte bedeuten eine heterogenere räumliche Verteilung der AT1-Zellen.

"""

st.markdown(
    parameter_text
)
