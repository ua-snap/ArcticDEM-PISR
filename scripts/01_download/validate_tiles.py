"""Validate downloaded tiles and build 3413 VRT mosaics.

Checks per tile: every expected asset is present; each raster opens with CRS EPSG:3413 and a 32 m pixel;
the DEM is float32 with nodata set; the full-resolution data and an overview read cleanly; the
valid-pixel fraction and DEM range are recorded. Also checks that DEM values agree in the overlap
strips between horizontal neighbours.

Outputs: qc/download_validation.csv, and dem/arcticdem_v<ver>_<res>m_{dem,datamask}_3413.vrt
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.windows import from_bounds

from arcticdem_pisr.config import Paths, load_config

gdal.UseExceptions()
RASTER_ASSETS = ["dem", "datamask", "count", "mad"]


def asset_path(raw, tile, res, ver, asset):
    return raw / tile / f"{tile}_{res}m_v{ver}_{asset}.tif"


def check_tile(tile, raw, res, ver):
    rec = {"tile": tile, "ok": True, "problems": ""}
    probs = []
    for asset in RASTER_ASSETS:
        p = asset_path(raw, tile, res, ver, asset)
        if not p.exists():
            probs.append(f"{asset}:missing")
            continue
        try:
            with rasterio.open(p) as ds:
                if ds.crs.to_epsg() != 3413:
                    probs.append(f"{asset}:crs={ds.crs}")
                if not np.allclose(ds.res, (res, res)):
                    probs.append(f"{asset}:res={ds.res}")
                a = ds.read(1, masked=True)
                if asset == "dem":
                    if ds.dtypes[0] != "float32" or ds.nodata is None:
                        probs.append(f"dem:dtype={ds.dtypes[0]},nodata={ds.nodata}")
                    if ds.overviews(1):
                        ds.read(1, out_shape=(ds.height // 8, ds.width // 8))
                    rec.update(width=ds.width, height=ds.height,
                               valid_frac=float(1 - np.ma.getmaskarray(a).mean()),
                               dem_min=float(a.min()) if a.count() else np.nan,
                               dem_max=float(a.max()) if a.count() else np.nan)
        except Exception as e:  # noqa: BLE001
            probs.append(f"{asset}:read_error:{e!r}")
    if not (raw / tile / f"{tile}_{res}m_v{ver}_meta.txt").exists():
        probs.append("meta:missing")
    rec["problems"] = ";".join(probs)
    rec["ok"] = not probs
    return rec


def overlap_diff(left, right, raw, res, ver):
    """Max |difference| of DEM values in the overlap strip between horizontally adjacent tiles."""
    pl, pr = asset_path(raw, left, res, ver, "dem"), asset_path(raw, right, res, ver, "dem")
    if not (pl.exists() and pr.exists()):
        return None
    with rasterio.open(pl) as a, rasterio.open(pr) as b:
        x0, x1 = max(a.bounds.left, b.bounds.left), min(a.bounds.right, b.bounds.right)
        y0, y1 = max(a.bounds.bottom, b.bounds.bottom), min(a.bounds.top, b.bounds.top)
        if x1 <= x0 or y1 <= y0:
            return None
        wa = a.read(1, window=from_bounds(x0, y0, x1, y1, a.transform), masked=True)
        wb = b.read(1, window=from_bounds(x0, y0, x1, y1, b.transform), masked=True)
        if wa.shape != wb.shape:
            return None
        d = np.ma.abs(wa - wb)
        return dict(left=left, right=right, n=int(d.count()), max_abs_diff=float(d.max()) if d.count() else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-overlap", type=int, default=30, help="neighbour pairs to compare")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]

    tl = pd.read_csv(paths.tile_list)
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(check_tile, tl.tile, [paths.raw] * len(tl), [res] * len(tl), [ver] * len(tl)))
    df = pd.DataFrame(recs)
    paths.qc.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.qc / "download_validation.csv", index=False)
    print(f"{df.ok.sum()}/{len(df)} tiles OK; {int((df.valid_frac == 0).sum())} tiles with no valid DEM data")
    if (~df.ok).any():
        print(df.loc[~df.ok, ["tile", "problems"]].to_string(index=False))

    tiles = set(tl.tile)
    pairs = [(t, f"{r:02d}_{c + 1:02d}") for t, r, c in zip(tl.tile, tl.row, tl.col)
             if f"{r:02d}_{c + 1:02d}" in tiles]
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(args.n_overlap, len(pairs)), replace=False)]
    od = [d for d in (overlap_diff(a, b, paths.raw, res, ver) for a, b in pairs) if d]
    od = pd.DataFrame(od)
    od.to_csv(paths.qc / "download_overlap_check.csv", index=False)
    if len(od):
        print(f"Overlap check: {len(od)} pairs, max |diff| = {od.max_abs_diff.max():.4f} m, "
              f"median of per-pair max = {od.max_abs_diff.median():.4f} m")

    # Each tile's outermost pixel ring disagrees with its neighbour (edge artifact), while the inner overlap
    # is identical. So crop every tile to its nominal 100 km footprint before mosaicking: no seams.
    paths.dem.mkdir(parents=True, exist_ok=True)
    ok = tl[tl.tile.isin(df.loc[df.ok, "tile"])]
    for asset in ("dem", "datamask"):
        cdir = paths.dem / f"tiles_cropped_{asset}"
        cdir.mkdir(exist_ok=True)
        files = []
        for r in ok.itertuples():
            src = asset_path(paths.raw, r.tile, res, ver, asset)
            with rasterio.open(src) as ds:
                b = ds.bounds
            win = [max(r.xmin, b.left), min(r.ymax, b.top), min(r.xmax, b.right), max(r.ymin, b.bottom)]
            dst = cdir / f"{r.tile}.vrt"
            gdal.Translate(str(dst), str(src), format="VRT", projWin=win)
            files.append(str(dst))
        vrt = paths.dem / f"arcticdem_v{ver}_{res}m_{asset}_3413.vrt"
        gdal.BuildVRT(str(vrt), files)
        print(f"Wrote {vrt} ({len(files)} tiles, cropped to nominal footprints)")


if __name__ == "__main__":
    main()
