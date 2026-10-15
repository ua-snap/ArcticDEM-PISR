"""Set PISR to NoData over water (Phase 2d).

ArcticDEM v4.1 mosaics flatten lakes and sea and flag them with datamask = 0. But datamask = 0 also covers DEM
voids (which we interpolate, see fill_voids.py) and some land that PGC filled from other sources (real slopes).
So water is defined as:
    datamask == 0  AND  valid in the UNFILLED 3338 DEM  AND  Horn slope < water_mask.max_slope_deg
(In a sample of 80 native tiles, 91% of datamask=0 cells have slope < 0.05°, versus 1.2% of datamask=1 land.)
Shoreline cells whose 3x3 window includes land keep their PISR.

Steps: warp the datamask VRT to the 3338 grid (nearest) -> water mask (uint8, 1 = water) -> apply to the PISR.
The unmasked PISR is kept as pisr_equinox_<res>m_3338_unmasked.tif. Stats -> qc/water_mask_stats.json.
"""
import argparse
import json
import time

import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal
from rasterio import features
from rasterio.windows import Window

from arcticdem_pisr.config import Paths, load_config

gdal.UseExceptions()
CO = ["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=ZSTD", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]


def horn_slope_deg(z, res):
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    p = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    q = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * res)
    return np.degrees(np.arctan(np.hypot(p, q)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--band-rows", type=int, default=2048)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
    max_slope = cfg["water_mask"]["max_slope_deg"]
    grid = json.loads(paths.target_grid.read_text())
    t0 = time.time()

    # 1. datamask on the 3338 grid (nearest; 0 is a real class here, so no source nodata)
    if not paths.datamask_3338.exists():
        tmp = paths.datamask_3338.with_suffix(".tmp.tif")
        gdal.Warp(str(tmp), str(paths.dem / f"arcticdem_v{ver}_{res}m_datamask_3413.vrt"), dstSRS=grid["crs"],
                  outputBounds=grid["bounds"], xRes=res, yRes=res, resampleAlg="near", srcNodata="None",
                  dstNodata=255, outputType=gdal.GDT_Byte, multithread=True,
                  warpOptions=["NUM_THREADS=ALL_CPUS"], creationOptions=CO + ["PREDICTOR=2"])
        tmp.rename(paths.datamask_3338)
        print(f"wrote {paths.datamask_3338}; {time.time() - t0:.0f}s", flush=True)

    # 2. keep the unmasked PISR, then write the masked PISR and the water mask band by band
    if not paths.pisr_3338_unmasked.exists():
        paths.pisr_3338.rename(paths.pisr_3338_unmasked)
    ak = gpd.read_file(paths.alaska_boundary).to_crs(grid["crs"])
    n = dict(water=0, water_in_ak=0, ak_cells=0, pisr_valid_before=0, pisr_valid_after=0, dm0_valid=0)
    with rasterio.open(paths.pisr_3338_unmasked) as src, rasterio.open(paths.dem_3338) as dem, \
            rasterio.open(paths.datamask_3338) as dm:
        prof = src.profile
        prof.update(bigtiff="YES", predictor=3, num_threads="ALL_CPUS")
        mprof = {**prof, "dtype": "uint8", "nodata": None, "predictor": 2}
        tmp_p = paths.pisr_3338.with_suffix(".tmp.tif")
        tmp_w = paths.water_mask.with_suffix(".tmp.tif")
        with rasterio.open(tmp_p, "w", **prof) as out_p, rasterio.open(tmp_w, "w", **mprof) as out_w:
            out_p.update_tags(**src.tags(), water_mask=f"NoData where ArcticDEM datamask=0, valid unfilled DEM, "
                                                       f"slope < {max_slope} deg")
            H, W = src.height, src.width
            for r0 in range(0, H, args.band_rows):
                r1 = min(r0 + args.band_rows, H)
                win = Window(0, r0, W, r1 - r0)
                z = dem.read(1, window=Window(-1, r0 - 1, W + 2, r1 - r0 + 2), boundless=True,
                             fill_value=np.nan).astype(np.float64)
                with np.errstate(invalid="ignore"):
                    flat = horn_slope_deg(z, res) < max_slope  # NaN window -> False
                water = (dm.read(1, window=win) == 0) & np.isfinite(z[1:-1, 1:-1]) & flat
                p = src.read(1, window=win)
                n["pisr_valid_before"] += int(np.isfinite(p).sum())
                n["dm0_valid"] += int(((dm.read(1, window=win) == 0) & np.isfinite(z[1:-1, 1:-1])).sum())
                p[water] = np.nan
                n["pisr_valid_after"] += int(np.isfinite(p).sum())
                akm = features.rasterize(ak.geometry, out_shape=p.shape, transform=src.window_transform(win)) == 1
                n["water"] += int(water.sum())
                n["water_in_ak"] += int((water & akm).sum())
                n["ak_cells"] += int(akm.sum())
                out_p.write(p, 1, window=win)
                out_w.write(water.astype(np.uint8), 1, window=win)
                print(f"rows {r0}-{r1}: water {water.mean():.3f}; {time.time() - t0:.0f}s", flush=True)
    for tmp, final in ((tmp_p, paths.pisr_3338), (tmp_w, paths.water_mask)):
        with rasterio.open(tmp, "r+") as d:
            d.build_overviews([4, 16, 64, 256], rasterio.enums.Resampling.average if d.dtypes[0] == "float32"
                              else rasterio.enums.Resampling.nearest)
        tmp.rename(final)
    cell_km2 = res * res / 1e6
    stats = {"max_slope_deg": max_slope, **n, "water_km2": n["water"] * cell_km2,
             "water_frac_of_dm0_valid": n["water"] / max(n["dm0_valid"], 1),
             "water_frac_of_ak_polygon": n["water_in_ak"] / max(n["ak_cells"], 1)}
    (paths.qc / "water_mask_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
