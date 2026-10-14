"""Fill small enclosed voids in the 32 m EPSG:3338 DEM by inverse-distance interpolation (Phase 2b).

A void is an 8-connected component of NaN cells. Components touching the raster edge or larger than
`void_fill.max_cells` (e.g. ocean, large data gaps) are left as NaN. Each small void is filled by IDW
(power `idw_power`) from its ring of valid neighbours (the 1-cell dilation of the void).

Outputs: <dem>_filled.tif (float32), <dem>_fillmask.tif (uint8: 1 = interpolated), and
qc/void_fill_stats.json with the void-size distribution.
"""
import argparse
import json
import time

import numpy as np
import rasterio
from scipy import ndimage

from arcticdem_pisr.config import Paths, load_config


def idw_fill(z, hole, power):
    """Fill `hole` cells of z (a small crop) by IDW from the valid ring around the hole."""
    ring = ndimage.binary_dilation(hole, structure=np.ones((3, 3), bool)) & ~hole & np.isfinite(z)
    hr, hc = np.nonzero(hole)
    rr, rc = np.nonzero(ring)
    if rr.size == 0:
        return False
    d2 = (hr[:, None] - rr[None, :]) ** 2 + (hc[:, None] - rc[None, :]) ** 2
    w = 1.0 / d2 ** (power / 2)
    z[hr, hc] = (w @ z[rr, rc]) / w.sum(axis=1)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    vf = cfg["void_fill"]
    t0 = time.time()

    with rasterio.open(paths.dem_3338) as src:
        profile = src.profile
        dem = src.read(1)
    print(f"read {dem.shape} in {time.time() - t0:.0f}s", flush=True)

    void = ~np.isfinite(dem)
    labels, n = ndimage.label(void, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    edge = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    small = (sizes > 0) & (sizes <= vf["max_cells"])
    small[edge] = False
    print(f"{n} void components; {small.sum()} small enclosed ({sizes[small].sum()} cells); "
          f"{time.time() - t0:.0f}s", flush=True)

    fillmask = np.zeros(dem.shape, np.uint8)
    objs = ndimage.find_objects(labels)
    n_fail = 0
    for lab in np.flatnonzero(small):
        sl = objs[lab - 1]
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        crop = dem[sl]
        hole = labels[sl] == lab
        if idw_fill(crop, hole, vf["idw_power"]):
            fillmask[sl][hole] = 1
        else:
            n_fail += 1
    del labels
    print(f"filled {int(fillmask.sum())} cells ({n_fail} components without a valid ring); "
          f"{time.time() - t0:.0f}s", flush=True)

    enclosed = sizes.copy()
    enclosed[edge] = 0
    s = enclosed[enclosed > 0]
    edges = [1, 10, 100, 1000, 10000, 100000, 1e12]
    stats = {
        "max_cells": vf["max_cells"], "idw_power": vf["idw_power"],
        "n_void_components": int(n), "n_enclosed": int(s.size),
        "n_filled_components": int(small.sum() - n_fail), "filled_cells": int(fillmask.sum()),
        "filled_area_km2": float(fillmask.sum() * abs(profile["transform"].a) ** 2 / 1e6),
        "enclosed_size_histogram_cells": {f"{int(lo)}-{int(hi) - 1 if hi < 1e12 else 'inf'}": int(((s >= lo) & (s < hi)).sum())
                                          for lo, hi in zip(edges[:-1], edges[1:])},
        "largest_enclosed_cells": int(s.max()) if s.size else 0,
    }
    paths.qc.mkdir(exist_ok=True)
    (paths.qc / "void_fill_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)

    profile.update(num_threads="ALL_CPUS")
    for path, arr, extra in ((paths.dem_3338_filled, dem, {}),
                             (paths.dem_fill_mask, fillmask, dict(dtype="uint8", nodata=None, predictor=2))):
        tmp = path.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **{**profile, **extra}) as dst:
            dst.write(arr, 1)
            if path == paths.dem_3338_filled:
                dst.update_tags(void_fill=json.dumps({k: stats[k] for k in ("max_cells", "idw_power", "filled_cells")}))
        with rasterio.open(tmp, "r+") as dst:
            dst.build_overviews([4, 16, 64, 256], rasterio.enums.Resampling.average if arr.dtype != np.uint8
                                else rasterio.enums.Resampling.max)
        tmp.rename(path)
        print(f"wrote {path}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
