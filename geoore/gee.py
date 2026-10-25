"""Optional real-data loader using Google Earth Engine.

Requires `pip install earthengine-api` and a one-time `earthengine authenticate`.
Downloads a cloud-masked Sentinel-2 median composite and SRTM DEM for a bounding
box, so the same pipeline runs on real ground.
"""
import io
import zipfile

import numpy as np

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def load_area(bbox, start="2023-01-01", end="2024-12-31", scale=30, project=None):
    """bbox = (min_lon, min_lat, max_lon, max_lat). Returns (bands, dem, transform, crs)."""
    import ee
    import rasterio
    import requests

    ee.Initialize(project=project)
    region = ee.Geometry.Rectangle(list(bbox))

    def mask(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(ok).divide(10000)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(start, end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
          .map(mask).select(S2_BANDS).median())
    dem = ee.Image("USGS/SRTMGL1_003").rename("DEM")
    img = s2.addBands(dem).clip(region).toFloat()

    url = img.getDownloadURL({"region": region, "scale": scale, "crs": "EPSG:3857",
                              "format": "GEO_TIFF"})
    data = requests.get(url, timeout=300).content
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
    with rasterio.MemoryFile(data) as mf, mf.open() as src:
        arr = src.read().astype("float32")
        transform, crs = src.transform, src.crs.to_string()
    bands = {b: np.nan_to_num(arr[i], nan=0.01) for i, b in enumerate(S2_BANDS)}
    return bands, np.nan_to_num(arr[-1]), transform, crs
