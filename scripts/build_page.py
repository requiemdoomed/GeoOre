"""Build the self-contained project page: docs/index.html.

Runs the pipeline, renders every layer as an image, collects real model
predictions and embeds it all (images, interactive map) into one HTML file,
so the page works on GitHub Pages and when opened straight from disk.
"""
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from geoore.grade_model import CLASSES, cross_validate, train_and_test  # noqa: E402
from geoore.indices import compute_indices, robust_scale  # noqa: E402
from geoore.mapviz import build_map  # noqa: E402
from geoore.prospectivity import score_area, top_targets  # noqa: E402
from geoore.synthetic import make_scene  # noqa: E402

VENDOR = ROOT / "scripts" / "vendor"
TEMPLATE = ROOT / "scripts" / "page_template.html"
OUT = ROOT / "docs" / "index.html"
SEED = 7


def to_uri(rgb, quality=84, size=640):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def cmap(arr, name):
    return matplotlib.colormaps[name](np.clip(arr, 0, 1))[..., :3]


def draw_overlay(rgb, scene, targets=None, faults=True):
    """Burn fault lines and targets into an image with matplotlib, no axes."""
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="bicubic")
    if faults:
        inv = ~scene.transform
        for f in scene.faults:
            xy = np.array([inv @ c for c in f.coords])
            ax.plot(xy[:, 0], xy[:, 1], color="#7fe3ee", lw=1.6, ls=(0, (5, 4)))
    if targets:
        for t in targets:
            ax.scatter(t["col"], t["row"], s=150, marker="*", c="#c6ff7a",
                       edgecolors="#10140f", linewidths=1)
            ax.text(t["col"] + 4, t["row"] - 3, str(t["rank"]), color="white", fontsize=11,
                    weight="bold", path_effects=[__import__("matplotlib.patheffects", fromlist=["x"])
                                                 .withStroke(linewidth=2.5, foreground="black")])
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB")) / 255.0


def hillshade(dem, az=315, alt=40):
    dy, dx = np.gradient(dem.astype(float))
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy) / 3)
    aspect = np.arctan2(-dx, dy)
    a, z = np.radians(az), np.radians(alt)
    return np.clip(np.sin(z) * np.sin(slope) + np.cos(z) * np.cos(slope) * np.cos(a - aspect), 0, 1)


def inline_map(html):
    """Swap the map's CDN links for an embedded copy of Leaflet.

    The GeoOre map only uses core Leaflet, so the extra libraries Folium links
    by default (Bootstrap, jQuery, Font Awesome, awesome-markers) are dropped.
    The page then works offline and never breaks because a CDN is down.
    """
    import re
    html = re.sub(r'<script src="https?://[^"]+"></script>\s*', "", html)
    html = re.sub(r'<link rel="stylesheet" href="https?://[^"]+"\s*/?>\s*', "", html)
    css = (VENDOR / "leaflet.css").read_text(encoding="utf-8")
    for name in ("layers-2x.png", "layers.png"):  # longer name first
        uri = "data:image/png;base64," + base64.b64encode((VENDOR / name).read_bytes()).decode()
        css = css.replace(f"images/{name}", uri)
    assets = ("<style>" + css + "</style>\n"
              "<script>" + (VENDOR / "leaflet.js").read_text(encoding="utf-8") + "</script>\n"
              # Folium builds popups with $(html)[0]; this is all of jQuery it needs
              "<script>window.$=function(h){var t=document.createElement('template');"
              "t.innerHTML=String(h).trim();return [t.content.firstElementChild];};</script>\n")
    return html.replace("<head>", "<head>\n" + assets, 1)


def main():
    scene = make_scene(size=200, seed=SEED)
    result = score_area(scene.bands, scene.dem, scene.faults, scene.transform, scene.samples)
    targets = top_targets(result, scene.transform, n=8)
    b = scene.bands
    idx = compute_indices(b)

    rgb = np.dstack([robust_scale(b["B4"]), robust_scale(b["B3"]), robust_scale(b["B2"])])
    shade = hillshade(scene.dem)[..., None]
    terrain = cmap(robust_scale(scene.dem), "gist_earth") * (0.45 + 0.55 * shade)

    layers = {
        "rgb": to_uri(rgb),
        "iron": to_uri(cmap(robust_scale(idx["iron_oxide"]), "YlOrRd")),
        "clay": to_uri(cmap(robust_scale(idx["clay"]), "PuBuGn")),
        "ndvi": to_uri(cmap(robust_scale(idx["ndvi"]), "YlGn")),
        "terrain": to_uri(draw_overlay(terrain, scene)),
        "score": to_uri(draw_overlay(cmap(result.score, "inferno"), scene, targets)),
        "conf": to_uri(cmap(result.confidence, "viridis")),
        "truth": to_uri(cmap(scene.ore_truth, "magma")),
        "score_plain": to_uri(cmap(result.score, "inferno")),
    }

    # Accuracy of the map across several random areas
    aucs = []
    for s in (7, 11, 23):
        sc = scene if s == SEED else make_scene(size=200, seed=s)
        r = result if s == SEED else score_area(sc.bands, sc.dem, sc.faults, sc.transform, sc.samples)
        aucs.append(roc_auc_score((sc.ore_truth > 0.3).ravel(), r.score.ravel()))

    model, metrics, _, test = train_and_test(scene.samples, seed=SEED)
    cv = cross_validate(scene.samples, seed=SEED)
    imp = model.global_importance(test).head(8)

    # Real predictions for a few held-out samples, one or two per class
    picks = []
    for g in ["high", "medium", "low"]:
        picks += list(test[test["grade"] == g].sample(2, random_state=SEED).index)
    samples = []
    for i in picks:
        row = test.loc[[i]]
        cls, contrib = model.explain_sample(row, top=3)
        proba = model.predict_proba(row)[0]
        samples.append({
            "id": row["sample_id"].iloc[0], "truth": row["grade"].iloc[0], "pred": cls,
            "proba": {c: round(float(p), 3) for c, p in zip(CLASSES, proba)},
            "assay": {k: round(float(row[k].iloc[0]), 2) for k in
                      ["Cu_ppm", "Au_ppb", "Fe_pct", "S_pct", "As_ppm", "Mo_ppm"]},
            "drivers": [[k, round(float(v), 3)] for k, v in contrib.items()],
        })

    fmap = build_map(result, scene.transform, scene.crs, targets, scene.samples, scene.faults,
                     grid_step=12)
    map_b64 = base64.b64encode(inline_map(fmap.get_root().render()).encode()).decode()

    data = {
        "layers": layers,
        "auc": [round(a, 3) for a in aucs],
        "acc": round(metrics["accuracy"] * 100, 1),
        "f1": round(metrics["macro_f1"], 3),
        "cv": round(float(cv.mean()), 3), "cv_sd": round(float(cv.std()), 3),
        "cm": metrics["confusion_matrix"], "classes": CLASSES,
        "n_test": int(len(test)), "n_cells": int(result.score.size),
        "importance": [[k, round(float(v), 3)] for k, v in imp.items()],
        "samples": samples,
        "targets": [{"rank": t["rank"], "score": round(t["score"], 2),
                     "conf": round(t["confidence"], 2), "why": t["explanation"]} for t in targets],
    }
    # Preview image for the README
    (ROOT / "docs" / "preview.jpg").write_bytes(base64.b64decode(layers["score"].split(",", 1)[1]))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data))
    html = html.replace("__MAP_B64__", map_b64)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
