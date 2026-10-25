import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from sklearn.metrics import roc_auc_score

from geoore.export import export_all
from geoore.grade_model import CLASSES, GradeClassifier, cross_validate, train_and_test
from geoore.indices import clay_ratio, compute_indices, iron_oxide_ratio, robust_scale
from geoore.pipeline import run
from geoore.prospectivity import explain_cell, score_area, top_targets
from geoore.synthetic import make_scene
from geoore.terrain import fault_distance, slope_degrees


@pytest.fixture(scope="module")
def scene():
    return make_scene(size=120, seed=3)


@pytest.fixture(scope="module")
def result(scene):
    return score_area(scene.bands, scene.dem, scene.faults, scene.transform, scene.samples)


def test_indices_basic():
    assert iron_oxide_ratio(np.array([0.2]), np.array([0.1]))[0] == pytest.approx(2, rel=1e-4)
    assert clay_ratio(np.array([0.3]), np.array([0.2]))[0] == pytest.approx(1.5, rel=1e-4)
    s = robust_scale(np.arange(100.0))
    assert s.min() == 0 and s.max() == 1


def test_indices_highlight_alteration(scene):
    idx = compute_indices(scene.bands)
    ore = scene.ore_truth > 0.3
    assert idx["iron_oxide"][ore].mean() > idx["iron_oxide"][~ore].mean()
    assert idx["clay"][ore].mean() > idx["clay"][~ore].mean()


def test_terrain(scene):
    assert slope_degrees(np.zeros((10, 10)), 30).max() == 0
    fd = fault_distance(scene.faults, scene.shape, scene.transform)
    assert fd.min() == 0 and np.isfinite(fd).all()


def test_prospectivity_finds_ore(scene, result):
    y = (scene.ore_truth > 0.3).ravel()
    assert roc_auc_score(y, result.score.ravel()) > 0.9
    assert result.score.shape == scene.shape
    assert ((result.confidence >= 0) & (result.confidence <= 1)).all()


def test_scoring_without_samples(scene):
    r = score_area(scene.bands, scene.dem, scene.faults, scene.transform, None)
    assert np.isfinite(r.score).all()


def test_targets_and_explanations(scene, result):
    t = top_targets(result, scene.transform, n=5)
    assert 1 <= len(t) <= 5
    assert t[0]["score"] >= t[-1]["score"]
    assert "prospectivity" in explain_cell(result, 0, 0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_grade_model_repeatable(seed):
    df = make_scene(size=120, seed=seed).samples
    model, metrics, _, test = train_and_test(df, seed=seed)
    assert metrics["accuracy"] > 0.85
    assert set(model.predict(test)) <= set(CLASSES)
    assert np.allclose(model.predict_proba(test).sum(1), 1, atol=1e-5)


def test_cross_validation(scene):
    assert cross_validate(scene.samples, folds=3).mean() > 0.8


def test_shap_and_save(scene, tmp_path):
    m = GradeClassifier().fit(scene.samples)
    imp = m.global_importance(scene.samples.head(100))
    assert imp.index[0].startswith(("log_Cu", "log_Au"))
    cls, contrib = m.explain_sample(scene.samples.head(1))
    assert cls in CLASSES and len(contrib) == 4
    m.save(tmp_path / "m.joblib")
    m2 = GradeClassifier.load(tmp_path / "m.joblib")
    assert (m2.predict(scene.samples) == m.predict(scene.samples)).all()


def test_exports(scene, result, tmp_path):
    t = top_targets(result, scene.transform, n=3)
    export_all(tmp_path, result, scene.transform, scene.crs, t, scene.samples)
    for name in ["zones", "targets", "samples"]:
        assert len(gpd.read_file(tmp_path / f"{name}.geojson")) > 0
        assert len(gpd.read_file(tmp_path / "shapefiles" / f"{name}.shp")) > 0
    with rasterio.open(tmp_path / "prospectivity.tif") as src:
        assert src.crs.to_string() == scene.crs and src.shape == scene.shape


def test_full_pipeline(tmp_path):
    *_, summary = run(tmp_path, seed=5, size=100)
    assert (tmp_path / "map.html").stat().st_size > 10_000
    assert json.loads((tmp_path / "summary.json").read_text())["grade_test_accuracy"] > 0.85
