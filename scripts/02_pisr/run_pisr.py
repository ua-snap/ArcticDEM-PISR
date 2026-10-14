"""Compute PISR over a DEM raster in row bands (Phase 2c). Restartable.

Works on any projected DEM (the projection geometry is handled per cell). Each band is read with a
1-cell halo (NaN outside the raster), processed by the numba kernel using all allotted threads, and written
into a tiled GeoTIFF on the same grid. Finished bands are logged to <out>.bands_done and skipped on restart.

Examples:
  run_pisr.py                                   # full run on the 32 m EPSG:3338 DEM
  run_pisr.py --bounds 150000 1150000 300000 1300000 --out .../pisr_pilot.tif     # sub-window (pilot)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from arcticdem_pisr.config import Paths, load_config
from arcticdem_pisr.solar import SolarParams, pisr

BLOCK = 512


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dem", default=None, help="input DEM (default: the 32 m EPSG:3338 DEM)")
    ap.add_argument("--out", default=None, help="output GeoTIFF (default: pisr/pisr_equinox_<res>m_3338.tif)")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="process only this sub-window (in the DEM's CRS), snapped to the grid")
    ap.add_argument("--band-rows", type=int, default=1024)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
    dem_path = Path(args.dem or paths.dem / f"arcticdem_v{ver}_{res}m_dem_3338.tif")
    out_path = Path(args.out or paths.pisr / f"pisr_equinox_{res}m_3338.tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = SolarParams(**cfg.get("pisr", {}))
    print(f"threads={os.environ.get('NUMBA_NUM_THREADS', 'default')} params={params}", flush=True)

    with rasterio.open(dem_path) as src:
        if args.bounds:
            w = from_bounds(*args.bounds, transform=src.transform).round_offsets().round_lengths()
            w = w.intersection(Window(0, 0, src.width, src.height))
        else:
            w = Window(0, 0, src.width, src.height)
        out_transform = src.window_transform(w)
        profile = dict(driver="GTiff", width=int(w.width), height=int(w.height), count=1, dtype="float32",
                       crs=src.crs, transform=out_transform, nodata=np.nan, tiled=True, blockxsize=BLOCK,
                       blockysize=BLOCK, compress="ZSTD", predictor=3, bigtiff="YES", num_threads="ALL_CPUS")
        done_log = out_path.with_name(out_path.name + ".bands_done")
        done = set(done_log.read_text().split()) if done_log.exists() and out_path.exists() else set()
        mode = "r+" if done else "w"
        if mode == "w" and done_log.exists():
            done_log.unlink()
        with rasterio.open(out_path, mode, **(profile if mode == "w" else {})) as dst:
            if mode == "w":
                dst.update_tags(**params.as_attrs(), source_dem=str(dem_path), units="kWh m-2 day-1",
                                description="Clear-sky PISR, idealized equinox (declination 0), local shading only")
            nrows = int(w.height)
            bands = [(r0, min(r0 + args.band_rows, nrows)) for r0 in range(0, nrows, args.band_rows)]
            t_all = time.time()
            for bi, (r0, r1) in enumerate(bands):
                key = f"{r0}-{r1}"
                if key in done:
                    continue
                t0 = time.time()
                rw = Window(w.col_off - 1, w.row_off + r0 - 1, w.width + 2, (r1 - r0) + 2)
                dem = src.read(1, window=rw, boundless=True, fill_value=np.nan, masked=True)
                dem = dem.astype(np.float64).filled(np.nan)
                band_window = Window(0, r0, w.width, r1 - r0)
                if np.isnan(dem[1:-1, 1:-1]).all():
                    out = np.full((r1 - r0, int(w.width)), np.nan, dtype=np.float32)
                else:
                    out = pisr(dem, dst.window_transform(band_window), src.crs, params)
                dst.write(out, 1, window=band_window)
                with open(done_log, "a") as f:
                    f.write(key + "\n")
                valid = np.isfinite(out)
                stats = (f"min={np.nanmin(out):.3f} max={np.nanmax(out):.3f}" if valid.any() else "empty")
                print(f"band {bi + 1}/{len(bands)} rows {key}: {valid.mean() * 100:5.1f}% valid, {stats}, "
                      f"{time.time() - t0:.1f}s", flush=True)
    print(f"Done in {(time.time() - t_all) / 60:.1f} min -> {out_path}")
    (out_path.with_name(out_path.name + ".params.json")).write_text(
        json.dumps({"dem": str(dem_path), "window": [w.col_off, w.row_off, w.width, w.height], **params.as_attrs()},
                   indent=2))


if __name__ == "__main__":
    main()
