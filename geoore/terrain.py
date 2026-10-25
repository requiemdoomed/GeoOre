"""Terrain and structural layers."""
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt


def slope_degrees(dem, pixel_size):
    dy, dx = np.gradient(dem.astype("float64"), pixel_size)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def roughness(dem, size=3):
    from scipy.ndimage import maximum_filter, minimum_filter
    return maximum_filter(dem, size) - minimum_filter(dem, size)


def fault_distance(faults, shape, transform):
    """Distance in metres from each cell to the nearest fault line."""
    if not faults:
        return np.full(shape, np.inf)
    mask = rasterize([(f, 1) for f in faults], out_shape=shape, transform=transform,
                     fill=0, all_touched=True, dtype="uint8")
    if mask.sum() == 0:
        return np.full(shape, 1e6)
    return distance_transform_edt(mask == 0) * abs(transform.a)
