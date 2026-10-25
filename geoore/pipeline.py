"""End-to-end run: score an area, train the grade model, export everything."""
import json
from pathlib import Path

import geopandas as gpd

from .export import export_all
from .grade_model import cross_validate, train_and_test
from .mapviz import build_map
from .prospectivity import score_area, top_targets
from .synthetic import make_scene


def run(out_dir="outputs", seed=7, size=200, faults_path=None, samples_path=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = make_scene(size=size, seed=seed)
    faults, samples = scene.faults, scene.samples
    if faults_path:
        faults = list(gpd.read_file(faults_path).to_crs(scene.crs).geometry)
    if samples_path:
        import pandas as pd
        samples = pd.read_csv(samples_path)

    result = score_area(scene.bands, scene.dem, faults, scene.transform, samples)
    targets = top_targets(result, scene.transform, n=8)

    model, metrics, _, test = train_and_test(samples, seed=seed)
    cv = cross_validate(samples, seed=seed)
    model.save(out / "grade_model.joblib")
    importance = model.global_importance(test)

    files = export_all(out, result, scene.transform, scene.crs, targets, samples)
    build_map(result, scene.transform, scene.crs, targets, samples, faults).save(str(out / "map.html"))

    summary = {
        "grade_test_accuracy": metrics["accuracy"],
        "grade_test_macro_f1": metrics["macro_f1"],
        "grade_cv_macro_f1_mean": float(cv.mean()),
        "grade_cv_macro_f1_std": float(cv.std()),
        "confusion_matrix(low,medium,high)": metrics["confusion_matrix"],
        "top_features": importance.head(6).round(4).to_dict(),
        "targets": [{k: t[k] for k in ("rank", "score", "confidence", "explanation")}
                    for t in targets],
        "files": [str(f) for f in files] + [str(out / "map.html")],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return scene, result, targets, model, summary
