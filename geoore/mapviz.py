"""Interactive Folium (Leaflet) map with heatmap, confidence and per-cell explanations."""
import base64
import io

import folium
import matplotlib
import numpy as np
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GRADE_COLORS = {"high": "#d7301f", "medium": "#fc8d59", "low": "#4575b4"}


def _png(arr, cmap="inferno", alpha=0.7):
    rgba = matplotlib.colormaps[cmap](np.clip(np.nan_to_num(arr), 0, 1))
    rgba[..., 3] = alpha
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _target_icon(rank):
    """Numbered target pin drawn with plain HTML, so no icon font is needed."""
    return (f'<div style="width:28px;height:28px;border-radius:50%;background:#b4472a;'
            f'border:2px solid #fff;color:#fff;font:700 13px sans-serif;display:flex;'
            f'align-items:center;justify-content:center;box-shadow:0 1px 4px rgba(0,0,0,.6)">'
            f'{rank}</div>')


def _bounds_latlon(shape, transform, crs):
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    h, w = shape
    xs, ys = [], []
    for c, r in [(0, 0), (w, 0), (0, h), (w, h)]:
        x, y = transform @ (c, r)
        lon, lat = tr.transform(x, y)
        xs.append(lon)
        ys.append(lat)
    return [[min(ys), min(xs)], [max(ys), max(xs)]], tr


def build_map(result, transform, crs, targets, samples=None, faults=None, grid_step=10):
    bounds, tr = _bounds_latlon(result.score.shape, transform, crs)
    centre = [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2]
    m = folium.Map(location=centre, zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)

    folium.raster_layers.ImageOverlay(_png(result.score, "inferno"), bounds=bounds,
                                      name="Prospectivity heatmap", show=True).add_to(m)
    folium.raster_layers.ImageOverlay(_png(result.confidence, "viridis", 0.6), bounds=bounds,
                                      name="Confidence", show=False).add_to(m)

    if faults:
        fg = folium.FeatureGroup("Faults")
        for f in faults:
            pts = [tr.transform(x, y)[::-1] for x, y in f.coords]
            folium.PolyLine(pts, color="#00e5ff", weight=2, dash_array="6").add_to(fg)
        fg.add_to(m)

    # Clickable explanation grid (sub-sampled so the page stays light)
    from .prospectivity import explain_cell
    grid = folium.FeatureGroup("Cell explanations (click)", show=False)
    h, w = result.score.shape
    for r in range(grid_step // 2, h, grid_step):
        for c in range(grid_step // 2, w, grid_step):
            lon, lat = tr.transform(*(transform @ (c + 0.5, r + 0.5)))
            folium.CircleMarker([lat, lon], radius=3, weight=0, fill=True, fill_opacity=0.5,
                                fill_color="#ffffff",
                                popup=folium.Popup(explain_cell(result, r, c), max_width=300)
                                ).add_to(grid)
    grid.add_to(m)

    if samples is not None:
        sg = folium.FeatureGroup("Assay samples", show=False)
        for _, s in samples.iterrows():
            lon, lat = tr.transform(s["x"], s["y"])
            folium.CircleMarker([lat, lon], radius=3, color=GRADE_COLORS[s["grade"]], fill=True,
                                popup=f"{s['sample_id']}: {s['grade']} ({s['CuEq_pct']:.2f}% CuEq)"
                                ).add_to(sg)
        sg.add_to(m)

    tg = folium.FeatureGroup("Drill targets")
    for t in targets:
        lon, lat = tr.transform(t["x"], t["y"])
        folium.Marker([lat, lon], tooltip=f"Target #{t['rank']}",
                      popup=folium.Popup(f"<b>Target #{t['rank']}</b><br>{t['explanation']}",
                                         max_width=320),
                      icon=folium.DivIcon(html=_target_icon(t["rank"]), icon_size=(28, 28),
                                          icon_anchor=(14, 14))).add_to(tg)
    tg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds(bounds)
    return m
