"""GeoOre Streamlit app. Run with:  streamlit run app.py"""
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from geoore.export import export_all
from geoore.grade_model import CLASSES, train_and_test
from geoore.mapviz import build_map
from geoore.prospectivity import DEFAULT_WEIGHTS, LABELS, score_area, top_targets
from geoore.synthetic import ELEMENTS, make_scene

st.set_page_config(page_title="GeoOre", page_icon="⛏️", layout="wide")
st.title("GeoOre: ore-field identification and grade classification")
st.caption("Satellite mineral indices + terrain + faults → prospectivity map. "
           "Assay geochemistry → high / medium / low grade.")

with st.sidebar:
    st.header("Study area")
    source = st.radio("Data source", ["Demo area (offline)", "Google Earth Engine"])
    seed = st.number_input("Demo seed", 0, 9999, 7)
    size = st.slider("Grid size (cells)", 100, 300, 200, 20)
    bbox = None
    if source == "Google Earth Engine":
        st.info("Needs `earthengine-api` and `earthengine authenticate`.")
        bbox_txt = st.text_input("Bounding box (min_lon,min_lat,max_lon,max_lat)",
                                 "-115.70,33.00,-115.60,33.08")
        gee_project = st.text_input("GEE cloud project ID", "")
        bbox = tuple(float(v) for v in bbox_txt.split(","))
    st.header("Evidence weights")
    weights = {k: st.slider(LABELS[k].capitalize(), 0.0, 1.0, v, 0.05)
               for k, v in DEFAULT_WEIGHTS.items()}
    blend = st.slider("Knowledge vs. model blend", 0.0, 1.0, 0.5, 0.05,
                      help="1 = expert weights only, 0 = sample-trained model only")
    threshold = st.slider("Prospect zone threshold", 0.3, 0.9, 0.6, 0.05)
    up_samples = st.file_uploader("Assay CSV (optional)", type="csv",
                                  help=f"Columns: x, y, grade, {', '.join(ELEMENTS)}")
    up_faults = st.file_uploader("Fault lines GeoJSON (optional)", type=["geojson", "json"])


@st.cache_data(show_spinner="Building study area…")
def load_demo(seed, size):
    return make_scene(size=size, seed=seed)


@st.cache_data(show_spinner="Downloading Sentinel-2 and SRTM from Earth Engine…")
def load_gee(bbox, project):
    from geoore.gee import load_area
    return load_area(bbox, project=project or None)


scene = load_demo(int(seed), size)
bands, dem, transform, crs, faults, samples = (scene.bands, scene.dem, scene.transform,
                                                scene.crs, scene.faults, scene.samples)
if source == "Google Earth Engine":
    try:
        bands, dem, transform, crs = load_gee(bbox, gee_project)
        faults, samples = [], None
    except Exception as e:  # keep the app usable without credentials
        st.error(f"Earth Engine load failed ({e}). Showing the demo area instead.")
if up_samples is not None:
    samples = pd.read_csv(up_samples)
if up_faults is not None:
    import geopandas as gpd
    faults = list(gpd.read_file(up_faults).to_crs(crs).geometry)

result = score_area(bands, dem, faults, transform, samples, weights, blend)
targets = top_targets(result, transform, n=8)

tab_map, tab_layers, tab_grade, tab_export = st.tabs(
    ["Prospectivity map", "Evidence layers", "Ore grade classifier", "Export"])

with tab_map:
    c1, c2, c3 = st.columns(3)
    c1.metric("Cells scored", f"{result.score.size:,}")
    c2.metric(f"Cells ≥ {threshold}", f"{(result.score >= threshold).mean():.1%}")
    c3.metric("Mean confidence", f"{result.confidence.mean():.2f}")
    m = build_map(result, transform, crs, targets, samples, faults)
    html = m.get_root().render()
    if hasattr(st, "iframe"):
        st.iframe(html, height=600)
    else:  # Streamlit < 1.52
        import streamlit.components.v1 as components
        components.html(html, height=600)
    st.subheader("Ranked sampling targets")
    st.dataframe(pd.DataFrame(targets)[["rank", "score", "confidence", "explanation"]],
                 hide_index=True, width="stretch")

with tab_layers:
    cols = st.columns(3)
    for i, (name, arr) in enumerate(list(result.layers.items()) + [("confidence", result.confidence)]):
        cols[i % 3].image(np.clip(arr, 0, 1), caption=LABELS.get(name, name),
                          width="stretch", clamp=True)

with tab_grade:
    if samples is None or "grade" not in samples:
        st.warning("Upload an assay CSV with a `grade` column to train the classifier.")
    else:
        model, metrics, _, test = train_and_test(samples)
        a, b = st.columns(2)
        a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
        b.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=[f"true {c}" for c in CLASSES],
                          columns=[f"pred {c}" for c in CLASSES])
        st.write("Confusion matrix (test set)")
        st.dataframe(cm)
        st.write("Which elements drive predictions (mean |SHAP|)")
        st.bar_chart(model.global_importance(test))
        st.subheader("Classify a sample")
        sid = st.selectbox("Sample", test["sample_id"] if "sample_id" in test else test.index)
        row = test[test["sample_id"] == sid] if "sample_id" in test else test.loc[[sid]]
        cls, contrib = model.explain_sample(row)
        proba = model.predict_proba(row)[0]
        st.write(f"**Predicted grade: {cls}** (true: {row['grade'].iloc[0]})")
        st.write({c: round(float(p), 3) for c, p in zip(CLASSES, proba)})
        st.write("Top drivers for this prediction (SHAP, positive pushes toward the predicted class):")
        st.dataframe(contrib.rename("shap").to_frame())

with tab_export:
    st.write("GeoTIFF rasters, GeoJSON and shapefiles open directly in QGIS or ArcGIS.")
    if st.button("Build export package"):
        with tempfile.TemporaryDirectory() as tmp:
            export_all(tmp, result, transform, crs, targets, samples, threshold)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in Path(tmp).rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(tmp))
        st.download_button("Download geoore_results.zip", buf.getvalue(),
                           "geoore_results.zip", "application/zip")
