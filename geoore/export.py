"""Export results to GeoJSON, shapefile and GeoTIFF for QGIS / ArcGIS."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import Point, shape


def raster_to_geotiff(arr, transform, crs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=-9999, compress="deflate") as dst:
        dst.write(np.nan_to_num(arr, nan=-9999).astype("float32"), 1)
    return path


def zones_gdf(score, confidence, transform, crs, threshold=0.6):
    """Polygonise cells above threshold into prospect zones."""
    mask = (score >= threshold).astype("uint8")
    recs = []
    for geom, val in shapes(mask, mask=mask.astype(bool), transform=transform):
        poly = shape(geom)
        recs.append({"geometry": poly, "area_ha": poly.area / 1e4})
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs) if recs else \
        gpd.GeoDataFrame({"area_ha": []}, geometry=[], crs=crs)
    if len(gdf):
        from rasterio.features import geometry_mask
        means, confs = [], []
        for g in gdf.geometry:
            m = ~geometry_mask([g], score.shape, transform)
            means.append(float(score[m].mean()))
            confs.append(float(confidence[m].mean()))
        gdf["mean_score"] = means
        gdf["mean_conf"] = confs
        gdf = gdf.sort_values("mean_score", ascending=False).reset_index(drop=True)
        gdf.insert(0, "zone_id", [f"Z{i + 1:02d}" for i in range(len(gdf))])
    return gdf


def targets_gdf(targets, crs):
    return gpd.GeoDataFrame(targets, geometry=[Point(t["x"], t["y"]) for t in targets], crs=crs)


def samples_gdf(df, crs):
    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)


def export_all(out_dir, result, transform, crs, targets, samples=None, threshold=0.6):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    files.append(raster_to_geotiff(result.score, transform, crs, out / "prospectivity.tif"))
    files.append(raster_to_geotiff(result.confidence, transform, crs, out / "confidence.tif"))
    layers = {
        "zones": zones_gdf(result.score, result.confidence, transform, crs, threshold),
        "targets": targets_gdf(targets, crs),
    }
    if samples is not None:
        layers["samples"] = samples_gdf(samples, crs)
    for name, gdf in layers.items():
        if not len(gdf):
            continue
        gj = out / f"{name}.geojson"
        gdf.to_crs("EPSG:4326").to_file(gj, driver="GeoJSON")
        shp_dir = out / "shapefiles"
        shp_dir.mkdir(exist_ok=True)
        shp = gdf.copy()
        # Shapefile field names are limited to 10 characters
        shp.columns = [c[:10] if c != "geometry" else c for c in shp.columns]
        shp.to_file(shp_dir / f"{name}.shp")
        files += [gj, shp_dir / f"{name}.shp"]
    return files
