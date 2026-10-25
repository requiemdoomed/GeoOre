"""Spectral mineral indices from Sentinel-2 / Landsat reflectance."""
import numpy as np

EPS = 1e-6


def iron_oxide_ratio(red, blue):
    """Red / Blue: ferric iron (hematite, goethite, gossans)."""
    return red / (blue + EPS)


def clay_ratio(swir1, swir2):
    """SWIR1 / SWIR2: Al-OH clays (kaolinite, illite, sericite)."""
    return swir1 / (swir2 + EPS)


def ferrous_ratio(swir1, nir):
    """SWIR1 / NIR: ferrous silicates."""
    return swir1 / (nir + EPS)


def ndvi(nir, red):
    return (nir - red) / (nir + red + EPS)


def compute_indices(bands):
    """bands: dict with B2,B3,B4,B8,B11,B12 arrays. Returns dict of index arrays."""
    return {
        "iron_oxide": iron_oxide_ratio(bands["B4"], bands["B2"]),
        "clay": clay_ratio(bands["B11"], bands["B12"]),
        "ferrous": ferrous_ratio(bands["B11"], bands["B8"]),
        "ndvi": ndvi(bands["B8"], bands["B4"]),
    }


def robust_scale(a, lo=2, hi=98):
    """Scale to 0-1 using percentiles so outliers do not dominate."""
    p_lo, p_hi = np.nanpercentile(a, [lo, hi])
    return np.clip((a - p_lo) / (p_hi - p_lo + EPS), 0, 1)
