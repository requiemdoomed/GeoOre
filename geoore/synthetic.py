"""Reproducible synthetic study area.

Generates a georeferenced multispectral scene (Sentinel-2 style bands), a DEM,
fault traces and geochemical assay samples so the full pipeline can run
offline. Hidden ore bodies drive alteration signatures and assay values, which
lets us check that the model recovers them.
"""
from dataclasses import dataclass, field

import numpy as np
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString

BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]  # blue, green, red, NIR, SWIR1, SWIR2
ELEMENTS = ["Cu_ppm", "Au_ppb", "Fe_pct", "S_pct", "As_ppm", "Zn_ppm", "Pb_ppm", "Mo_ppm"]


@dataclass
class Scene:
    bands: dict
    dem: np.ndarray
    faults: list
    transform: object
    crs: str
    ore_truth: np.ndarray
    samples: object = field(default=None)

    @property
    def shape(self):
        return self.dem.shape


def _smooth_noise(rng, shape, sigma):
    return gaussian_filter(rng.normal(size=shape), sigma)


def make_scene(size=200, n_bodies=4, seed=7, pixel=30.0, origin=(500000.0, 7000000.0),
               crs="EPSG:32633"):
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]

    # Terrain: broad ridges plus fine relief
    dem = 400 + 120 * _smooth_noise(rng, (h, w), 25) / 0.05 * 0.05
    dem += 40 * np.sin(xx / 23.0) * np.cos(yy / 31.0) + 15 * _smooth_noise(rng, (h, w), 4)

    # Faults: random straight-ish traces
    faults = []
    for _ in range(3):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, np.pi)
        L = size * 1.5
        p0 = (x0 - L * np.cos(ang), y0 - L * np.sin(ang))
        p1 = (x0 + L * np.cos(ang), y0 + L * np.sin(ang))
        faults.append((p0, p1))

    # Ore bodies sit near faults (structural control)
    ore = np.zeros((h, w))
    for i in range(n_bodies):
        (fx0, fy0), (fx1, fy1) = faults[i % len(faults)]
        t = rng.uniform(0.35, 0.65)
        cx = np.clip(fx0 + t * (fx1 - fx0) + rng.normal(0, 4), 15, w - 15)
        cy = np.clip(fy0 + t * (fy1 - fy0) + rng.normal(0, 4), 15, h - 15)
        r = rng.uniform(6, 12)
        ore += rng.uniform(0.7, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    ore = np.clip(ore, 0, 1)

    # Alteration halos: iron oxide (gossan) and clay (argillic), wider than the ore
    halo = gaussian_filter(ore, 4)
    halo /= halo.max() + 1e-9
    iron = np.clip(halo + 0.08 * _smooth_noise(rng, (h, w), 3) / 0.1, 0, 1)
    clay = np.clip(gaussian_filter(ore, 6) / (gaussian_filter(ore, 6).max() + 1e-9)
                   + 0.08 * _smooth_noise(rng, (h, w), 3) / 0.1, 0, 1)
    veg = np.clip(0.5 + 0.6 * _smooth_noise(rng, (h, w), 10) / 0.03 * 0.05, 0, 1) * (1 - 0.6 * halo)

    base = 0.12 + 0.03 * _smooth_noise(rng, (h, w), 8) / 0.03 * 0.3
    noise = lambda: 0.006 * rng.normal(size=(h, w))
    bands = {
        "B2": base * 0.8 - 0.02 * iron + noise(),
        "B3": base * 0.95 + 0.03 * veg + noise(),
        "B4": base * 1.05 + 0.10 * iron - 0.03 * veg + noise(),
        "B8": base * 1.3 + 0.25 * veg + noise(),
        "B11": base * 1.6 + 0.06 * clay + noise(),
        "B12": base * 1.3 - 0.07 * clay + noise(),
    }
    bands = {k: np.clip(v, 0.005, 1.0).astype("float32") for k, v in bands.items()}

    transform = from_origin(origin[0], origin[1], pixel, pixel)
    fault_lines = [LineString([transform @ p0, transform @ p1]) for p0, p1 in faults]
    scene = Scene(bands=bands, dem=dem.astype("float32"), faults=fault_lines,
                  transform=transform, crs=crs, ore_truth=ore.astype("float32"))
    scene.samples = make_samples(scene, rng=rng)
    return scene


def make_samples(scene, n=900, rng=None):
    """Geochemical samples whose grade depends on the hidden ore field."""
    import pandas as pd

    rng = rng or np.random.default_rng(0)
    h, w = scene.shape
    # Bias half the samples toward mineralised ground, as real campaigns do
    rows = np.concatenate([rng.integers(0, h, n // 2),
                           np.zeros(n - n // 2, dtype=int)])
    cols = np.concatenate([rng.integers(0, w, n // 2),
                           np.zeros(n - n // 2, dtype=int)])
    hot = np.argwhere(scene.ore_truth > 0.15)
    pick = hot[rng.integers(0, len(hot), n - n // 2)]
    rows[n // 2:], cols[n // 2:] = pick[:, 0], pick[:, 1]

    o = scene.ore_truth[rows, cols] + rng.normal(0, 0.07, n)
    lg = lambda mu, s: np.exp(rng.normal(mu, s, n))
    df = pd.DataFrame({
        "Cu_ppm": lg(3.5, 0.5) * (1 + 40 * np.clip(o, 0, None) ** 1.5),
        "Au_ppb": lg(1.5, 0.6) * (1 + 60 * np.clip(o, 0, None) ** 2),
        "Fe_pct": np.clip(rng.normal(3.5, 0.8, n) + 6 * o, 0.3, None),
        "S_pct": np.clip(lg(-2.5, 0.5) * (1 + 25 * np.clip(o, 0, None)), 0, None),
        "As_ppm": lg(2.0, 0.7) * (1 + 8 * np.clip(o, 0, None)),
        "Zn_ppm": lg(4.2, 0.4) * (1 + 1.5 * np.clip(o, 0, None)),
        "Pb_ppm": lg(2.8, 0.5),
        "Mo_ppm": lg(0.5, 0.6) * (1 + 10 * np.clip(o, 0, None) ** 1.2),
    })
    # Grade from a copper-equivalent value, so labels follow the metals
    cueq = df["Cu_ppm"] / 1e4 + df["Au_ppb"] / 1e3 * 0.6  # % Cu-eq (simplified)
    df["CuEq_pct"] = cueq
    df["grade"] = np.select([cueq >= 0.15, cueq >= 0.05], ["high", "medium"], "low")
    x, y = scene.transform @ (cols + 0.5, rows + 0.5)
    df.insert(0, "y", y)
    df.insert(0, "x", x)
    df.insert(0, "sample_id", [f"GS-{i:04d}" for i in range(n)])
    return df
