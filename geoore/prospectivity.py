"""Cell-level prospectivity scoring.

Two layers of evidence are combined:
1. A knowledge-driven fuzzy overlay (weighted indices + terrain + faults).
2. A data-driven random forest trained on cells where samples returned
   high/medium grade (positives) versus low grade (negatives).
The final score blends both; confidence comes from agreement between the
bootstrapped forests.
"""
from dataclasses import dataclass

import numpy as np
from rasterio.transform import rowcol
from sklearn.ensemble import RandomForestClassifier

from .indices import compute_indices, robust_scale
from .terrain import fault_distance, roughness, slope_degrees

DEFAULT_WEIGHTS = {"iron_oxide": 0.30, "clay": 0.25, "fault_proximity": 0.25,
                   "low_vegetation": 0.10, "relief": 0.10}

LABELS = {"iron_oxide": "iron-oxide alteration", "clay": "clay alteration",
          "fault_proximity": "near a fault", "low_vegetation": "exposed ground",
          "relief": "rugged terrain"}


@dataclass
class ProspectivityResult:
    score: np.ndarray
    knowledge: np.ndarray
    model: np.ndarray
    confidence: np.ndarray
    layers: dict
    contributions: dict
    fault_dist_m: np.ndarray


def evidence_layers(bands, dem, faults, transform, fault_scale_m=600.0):
    idx = compute_indices(bands)
    fd = fault_distance(faults, dem.shape, transform)
    layers = {
        "iron_oxide": robust_scale(idx["iron_oxide"]),
        "clay": robust_scale(idx["clay"]),
        "fault_proximity": np.exp(-fd / fault_scale_m),
        "low_vegetation": 1 - robust_scale(idx["ndvi"]),
        "relief": robust_scale(roughness(dem) + 0.2 * slope_degrees(dem, abs(transform.a))),
    }
    return layers, fd


def knowledge_score(layers, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.values())
    contrib = {k: layers[k] * w / total for k, w in weights.items()}
    return sum(contrib.values()), contrib


def _stack(layers):
    names = list(DEFAULT_WEIGHTS)
    return names, np.stack([layers[n] for n in names], axis=-1)


def sample_training_set(layers, samples, transform):
    names, cube = _stack(layers)
    h, w = cube.shape[:2]
    r, c = rowcol(transform, samples["x"].values, samples["y"].values)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    X = cube[r[ok], c[ok]]
    y = (samples["grade"].values[ok] != "low").astype(int)
    return X, y, names


def model_score(layers, samples, transform, n_models=5, seed=0):
    X, y, names = sample_training_set(layers, samples, transform)
    _, cube = _stack(layers)
    flat = cube.reshape(-1, cube.shape[-1])
    rng = np.random.default_rng(seed)
    preds = []
    for i in range(n_models):
        boot = rng.integers(0, len(X), len(X))
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                    random_state=seed + i, n_jobs=-1)
        rf.fit(X[boot], y[boot])
        preds.append(rf.predict_proba(flat)[:, 1])
    preds = np.array(preds)
    mean = preds.mean(0).reshape(cube.shape[:2])
    std = preds.std(0).reshape(cube.shape[:2])
    return mean, std


def score_area(bands, dem, faults, transform, samples=None, weights=None, blend=0.5):
    layers, fd = evidence_layers(bands, dem, faults, transform)
    know, contrib = knowledge_score(layers, weights)
    know = robust_scale(know, 1, 99)
    if samples is not None and len(samples) > 20 and samples["grade"].nunique() > 1:
        model, std = model_score(layers, samples, transform)
        score = blend * know + (1 - blend) * model
        # Confidence: low bootstrap spread and agreement between both methods
        confidence = np.clip(1 - 2 * std - 0.5 * np.abs(know - model), 0, 1)
    else:
        model = np.full_like(know, np.nan)
        score = know
        confidence = np.full_like(know, 0.5)  # no ground truth to check against
    return ProspectivityResult(score=score, knowledge=know, model=model,
                               confidence=confidence, layers=layers,
                               contributions=contrib, fault_dist_m=fd)


def explain_cell(result, row, col, top=3):
    """Plain-language reason for a cell's score."""
    parts = sorted(((k, float(v[row, col])) for k, v in result.contributions.items()),
                   key=lambda kv: kv[1], reverse=True)
    s = float(result.score[row, col])
    level = "High" if s >= 0.6 else "Moderate" if s >= 0.35 else "Low"
    drivers = ", ".join(f"{LABELS[k]} ({v:.2f})" for k, v in parts[:top])
    fd = float(result.fault_dist_m[row, col])
    return (f"{level} prospectivity ({s:.2f}, confidence {result.confidence[row, col]:.2f}). "
            f"Main drivers: {drivers}. Nearest fault {fd:,.0f} m.")


def top_targets(result, transform, n=10, min_sep=10):
    """Pick the n best local maxima, at least min_sep cells apart."""
    s = result.score.copy()
    out = []
    for _ in range(n):
        r, c = np.unravel_index(np.nanargmax(s), s.shape)
        if not np.isfinite(s[r, c]) or s[r, c] <= 0:
            break
        x, y = transform @ (c + 0.5, r + 0.5)
        out.append({"rank": len(out) + 1, "row": int(r), "col": int(c), "x": x, "y": y,
                    "score": float(result.score[r, c]),
                    "confidence": float(result.confidence[r, c]),
                    "explanation": explain_cell(result, r, c)})
        s[max(0, r - min_sep):r + min_sep + 1, max(0, c - min_sep):c + min_sep + 1] = -1
    return out
