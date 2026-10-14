"""Real-terrain grid-invariance check: PISR computed on the 3338 DEM vs PISR computed natively in 3413.

The 3413 result is warped (bilinear) onto the 3338 pilot grid. If projection geometry is handled correctly,
the bias is ~0 and the differences are resampling noise with no dependence on aspect. A convergence error
would show up as a strong aspect-dependent bias, since the grids are rotated ~105° relative to each other.
Writes a CSV of stats and a figure.
"""
import argparse
import json

import matplotlib
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def hillshade(z, res, az=315, alt=45):
    gy, gx = np.gradient(z, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, h = np.deg2rad(az), np.deg2rad(alt)
    return np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pisr-3338", required=True)
    ap.add_argument("--pisr-3413", required=True)
    ap.add_argument("--dem-3338", required=True)
    ap.add_argument("--out-prefix", required=True, help="path prefix for .json stats and .png figure")
    args = ap.parse_args()

    with rasterio.open(args.pisr_3338) as a:
        p38 = a.read(1)
        tr, crs, shape = a.transform, a.crs, a.shape
        bounds = a.bounds
    p13 = np.full(shape, np.nan, np.float32)
    with rasterio.open(args.pisr_3413) as b:
        reproject(rasterio.band(b, 1), p13, dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
    with rasterio.open(args.dem_3338) as d:
        dem = d.read(1, window=rasterio.windows.from_bounds(*bounds, transform=d.transform))

    # Aspect (true, approx: 3338 convergence is small here) and slope from the 3338 DEM, for stratifying.
    gy, gx = np.gradient(dem.astype(np.float64), tr.a)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, gy)) + 180) % 360  # downslope azimuth, 0 = north-facing

    # Trim edges (halo/resampling effects) and require valid data in both.
    m = np.isfinite(p38) & np.isfinite(p13)
    m[:5, :] = m[-5:, :] = False
    m[:, :5] = m[:, -5:] = False
    d = (p38 - p13)[m]
    stats = dict(n=int(m.sum()), mean_3338=float(p38[m].mean()), mean_3413_warped=float(p13[m].mean()),
                 bias=float(d.mean()), rel_bias=float(d.mean() / p38[m].mean()), rmse=float(np.sqrt((d ** 2).mean())),
                 mae=float(np.abs(d).mean()), r=float(np.corrcoef(p38[m], p13[m])[0, 1]))
    by_aspect = {}
    for lo in range(0, 360, 45):
        sel = m & (slope > 10) & (((aspect - lo - 22.5) % 360) < 45)
        if sel.sum() > 100:
            by_aspect[f"{lo:03d}"] = float((p38 - p13)[sel].mean())
    stats["bias_by_aspect_slope_gt10"] = by_aspect
    print(json.dumps(stats, indent=2))
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(stats, f, indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax[0].imshow(hillshade(dem, tr.a), cmap="gray", extent=ext)
    ax[0].set_title("Hillshade (3338 DEM)")
    im = ax[1].imshow(p38, cmap="magma", extent=ext, vmin=np.nanpercentile(p38, 1), vmax=np.nanpercentile(p38, 99))
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="kWh m$^{-2}$ day$^{-1}$")
    ax[1].set_title("PISR computed on 3338 grid")
    lim = np.nanpercentile(np.abs(d), 99)
    im = ax[2].imshow(np.where(m, p38 - p13, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    fig.colorbar(im, ax=ax[2], shrink=0.8, label="kWh m$^{-2}$ day$^{-1}$")
    ax[2].set_title(f"3338 minus (3413→3338)\nbias {stats['bias']:+.4f}, RMSE {stats['rmse']:.3f}, r {stats['r']:.4f}")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.savefig(args.out_prefix + ".png", dpi=100)
    print(args.out_prefix + ".png")


if __name__ == "__main__":
    main()
