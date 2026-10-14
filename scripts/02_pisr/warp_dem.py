"""Warp the 3413 DEM mosaic VRT onto the final EPSG:3338 target grid (Phase 2a).

Cubic resampling (smooth field; preserves slopes better than bilinear). Output: tiled BigTIFF, float32,
ZSTD + floating-point predictor, NaN nodata, with overviews for quick-looks.
"""
import argparse
import json

from osgeo import gdal

from arcticdem_pisr.config import Paths, load_config

gdal.UseExceptions()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--resampling", default="cubic")
    ap.add_argument("--threads", default="ALL_CPUS")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
    grid = json.loads(paths.target_grid.read_text())

    src = paths.dem / f"arcticdem_v{ver}_{res}m_dem_3413.vrt"
    dst = paths.dem / f"arcticdem_v{ver}_{res}m_dem_3338.tif"
    tmp = dst.with_suffix(".tmp.tif")
    gdal.SetConfigOption("GDAL_CACHEMAX", "8192")
    gdal.Warp(
        str(tmp), str(src),
        dstSRS=grid["crs"], outputBounds=grid["bounds"], xRes=grid["resolution_m"], yRes=grid["resolution_m"],
        width=0, height=0, resampleAlg=args.resampling, errorThreshold=0.05, outputType=gdal.GDT_Float32,
        dstNodata=float("nan"), multithread=True, warpMemoryLimit=4096,
        warpOptions=[f"NUM_THREADS={args.threads}", "INIT_DEST=NO_DATA"],
        creationOptions=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=ZSTD", "PREDICTOR=3",
                         "BIGTIFF=YES", f"NUM_THREADS={args.threads}"],
    )
    ds = gdal.Open(str(tmp))
    assert (ds.RasterXSize, ds.RasterYSize) == (grid["width"], grid["height"]), "grid mismatch"
    assert list(ds.GetGeoTransform()) == [grid["bounds"][0], res, 0, grid["bounds"][3], 0, -res]
    ds = None
    gdal.Open(str(tmp), gdal.GA_Update).BuildOverviews("AVERAGE", [4, 16, 64, 256])
    tmp.rename(dst)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
