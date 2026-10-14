"""Quick-look coverage map: decimated DEM + datamask, with the 3338 product bbox and Alaska boundary (in 3413)."""
import geopandas as gpd
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arcticdem_pisr.config import Paths, load_config  # noqa: E402

cfg = load_config()
paths = Paths(cfg)
res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
dec = 50
with rasterio.open(paths.dem / f"arcticdem_v{ver}_{res}m_dem_3413.vrt") as ds:
    dem = ds.read(1, out_shape=(ds.height // dec, ds.width // dec), masked=True)
    ext = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]
fig, ax = plt.subplots(figsize=(9, 11))
ax.set_facecolor("#dfe7ee")
im = ax.imshow(dem, extent=ext, cmap="terrain", vmin=0, vmax=4000)
gpd.read_file(paths.alaska_boundary).to_crs(3413).boundary.plot(ax=ax, color="k", lw=0.4)
gpd.read_file(paths.aoi_3338).to_crs(3413).boundary.plot(ax=ax, color="crimson", lw=1.2)
gpd.read_file(paths.aoi_3413).boundary.plot(ax=ax, color="crimson", lw=0.6, ls="--")
ax.set_title(f"ArcticDEM v{ver} {res} m: downloaded tiles (EPSG:3413)\n"
             f"red = product bbox (EPSG:3338); dashed = download AOI; valid {100 * (1 - dem.mask.mean()):.1f}% of VRT extent")
fig.colorbar(im, ax=ax, shrink=0.6, label="Elevation (m)")
out = paths.figures / "qc_download_coverage.png"
paths.figures.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print(out)
