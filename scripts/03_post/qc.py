"""QC of the full PISR raster (Phase 3b). Writes qc/qc_report.json, CSVs, and qc/qc_samples.npz for figures.

Checks:
  1. Grid: CRS/size/transform equal target_grid.json and the 3338 DEM.
  2. Completeness (full resolution): PISR is valid exactly where the DEM's 3x3 window is fully valid.
  3. Range: 0 <= PISR <= upper bound (flat surface at the equator, same atmosphere).
  4. Reproducibility: random windows recomputed from the DEM match the file (to 1e-4; the geometry lattice is
     anchored differently per window, so float32 rounding differs).
  5. Physics on real terrain (sampled windows): flat cells match the flat reference curve vs latitude;
     mean PISR by true aspect is S > E ≈ W > N on steeper slopes.
  6. Seams: no step at processing band boundaries (row differences across boundaries vs nearby rows).
Also builds overviews on the PISR GeoTIFF if missing (for figures).
"""
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio import features
from rasterio.windows import Window
from scipy import ndimage

from arcticdem_pisr.config import Paths, load_config
from arcticdem_pisr.solar import SolarParams, lat_and_jacobian, pisr, reference_flat

gdal.UseExceptions()
HIST_EDGES = np.round(np.arange(0, 8.0001, 0.02), 4)


def dem_window(D, P, win):
    """The DEM window covering PISR window `win` (same grid spacing/alignment assumed)."""
    x, y = P.transform * (win.col_off, win.row_off)
    c, r = ~D.transform * (x, y)
    return Window(int(round(c)), int(round(r)), win.width, win.height)


def horn_true(dem_halo, transform, crs):
    """True slope (deg) and true downslope azimuth (deg from north) for the interior of dem_halo."""
    z = dem_halo
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dx, dy = abs(transform.a), abs(transform.e)
    p = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    q = ((a + 2 * b + c) - (g + 2 * h + i)) / (8 * dy)
    lat, jxe, jye, jxn, jyn = lat_and_jacobian(transform, p.shape, crs)
    ge, gn = p * jxe + q * jye, p * jxn + q * jyn
    slope = np.degrees(np.arctan(np.hypot(ge, gn)))
    aspect = np.degrees(np.arctan2(-ge, -gn)) % 360
    return slope, aspect, np.degrees(lat), (lat, jxe, jye, jxn, jyn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--band-rows", type=int, default=2048)
    ap.add_argument("--n-windows", type=int, default=300)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--pisr", default=None, help="override PISR path (e.g. a pilot)")
    ap.add_argument("--dem", default=None, help="override DEM path")
    ap.add_argument("--qc-dir", default=None, help="override output dir")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
    params = SolarParams(**cfg.get("pisr", {}))
    dem_path = args.dem or paths.dem_3338_filled
    pisr_path = args.pisr or paths.pisr / f"pisr_equinox_{res}m_3338.tif"
    if args.qc_dir:
        paths.qc = __import__("pathlib").Path(args.qc_dir)
    grid = json.loads(paths.target_grid.read_text())
    report = {"pisr_file": str(pisr_path), "dem_file": str(dem_path), "params": params.as_attrs()}
    paths.qc.mkdir(exist_ok=True)

    # --- 1. grid -----------------------------------------------------------------------------------
    with rasterio.open(pisr_path) as P, rasterio.open(dem_path) as D:
        g = dict(crs_ok=P.crs.to_epsg() == 3338, size_ok=(P.width, P.height) == (grid["width"], grid["height"]),
                 transform_ok=list(P.transform)[:6] == grid["transform"], matches_dem=P.transform == D.transform
                 and P.shape == D.shape, dtype=P.dtypes[0], nodata=str(P.nodata), overviews=P.overviews(1))
    report["grid"] = g
    if not g["overviews"]:
        gdal.Open(str(pisr_path), gdal.GA_Update).BuildOverviews("AVERAGE", [4, 16, 64, 256])
        report["grid"]["overviews_built"] = True
    print("grid:", g, flush=True)

    # --- 2/3. completeness, range, histograms (full resolution) -----------------------------------
    ak = gpd.read_file(paths.alaska_boundary).to_crs(3338)
    bound = float(reference_flat([0.0], params)[0])
    cnt = dict(cells=0, dem_valid=0, dem_interior_valid=0, pisr_valid=0, nan_where_expected_valid=0,
               valid_where_dem_nan=0, above_bound=0, below_zero=0, ak_cells=0, ak_dem_valid=0, ak_pisr_valid=0)
    hist_all = np.zeros(len(HIST_EDGES) - 1, np.int64)
    hist_ak = np.zeros_like(hist_all)
    vmin, vmax = np.inf, -np.inf
    with rasterio.open(pisr_path) as P, rasterio.open(dem_path) as D:
        H, W = P.height, P.width
        for r0 in range(0, H, args.band_rows):
            r1 = min(r0 + args.band_rows, H)
            dem = D.read(1, window=dem_window(D, P, Window(-1, r0 - 1, W + 2, r1 - r0 + 2)), boundless=True,
                         fill_value=np.nan)
            dv = np.isfinite(dem)
            interior = ndimage.minimum_filter(dv.astype(np.uint8), size=3, mode="constant", cval=0)[1:-1, 1:-1] == 1
            dv = dv[1:-1, 1:-1]
            pv = P.read(1, window=Window(0, r0, W, r1 - r0))
            pvalid = np.isfinite(pv)
            akm = features.rasterize(ak.geometry, out_shape=pv.shape,
                                     transform=P.window_transform(Window(0, r0, W, r1 - r0))) == 1
            cnt["cells"] += pv.size
            cnt["dem_valid"] += int(dv.sum())
            cnt["dem_interior_valid"] += int(interior.sum())
            cnt["pisr_valid"] += int(pvalid.sum())
            cnt["nan_where_expected_valid"] += int((interior & ~pvalid).sum())
            cnt["valid_where_dem_nan"] += int((pvalid & ~dv).sum())
            cnt["ak_cells"] += int(akm.sum())
            cnt["ak_dem_valid"] += int((akm & dv).sum())
            cnt["ak_pisr_valid"] += int((akm & pvalid).sum())
            vals = pv[pvalid]
            if vals.size:
                vmin, vmax = min(vmin, float(vals.min())), max(vmax, float(vals.max()))
                cnt["above_bound"] += int((vals > bound * (1 + 1e-6)).sum())
                cnt["below_zero"] += int((vals < 0).sum())
                hist_all += np.histogram(vals, HIST_EDGES)[0]
                hist_ak += np.histogram(pv[pvalid & akm], HIST_EDGES)[0]
            print(f"rows {r0}-{r1}: valid {pvalid.mean():.3f}", flush=True)
    cnt.update(min=vmin, max=vmax, upper_bound=bound,
               pisr_valid_frac_of_bbox=cnt["pisr_valid"] / cnt["cells"],
               ak_pisr_valid_frac=cnt["ak_pisr_valid"] / max(cnt["ak_cells"], 1))
    report["completeness_range"] = cnt
    pd.DataFrame({"bin_lo": HIST_EDGES[:-1], "bin_hi": HIST_EDGES[1:], "count_bbox": hist_all,
                  "count_alaska": hist_ak}).to_csv(paths.qc / "pisr_histogram.csv", index=False)
    print("completeness/range:", json.dumps(cnt, indent=1), flush=True)

    # --- 4/5. sampled windows: reproducibility + physics -------------------------------------------
    rng = np.random.default_rng(42)
    samples = {k: [] for k in ("lat", "slope", "aspect", "pisr", "elev", "x", "y")}
    repro_max = 0.0
    n_done, tries = 0, 0
    w = args.win
    with rasterio.open(pisr_path) as P, rasterio.open(dem_path) as D:
        while n_done < args.n_windows and tries < 50 * args.n_windows:
            tries += 1
            r0, c0 = int(rng.integers(0, P.height - w)), int(rng.integers(0, P.width - w))
            win = Window(c0, r0, w, w)
            pv = P.read(1, window=win)
            if np.isfinite(pv).mean() < 0.5:
                continue
            dem = D.read(1, window=dem_window(D, P, Window(c0 - 1, r0 - 1, w + 2, w + 2)), boundless=True,
                         fill_value=np.nan).astype(np.float64)
            tr = P.window_transform(win)
            slope, aspect, lat, geo = horn_true(dem, tr, P.crs)
            fresh = pisr(dem, tr, P.crs, params, geo=geo)
            both = np.isfinite(fresh) & np.isfinite(pv)
            assert (np.isfinite(fresh) == np.isfinite(pv)).all(), "NaN pattern differs from fresh computation"
            repro_max = max(repro_max, float(np.abs(fresh[both] - pv[both]).max()) if both.any() else 0.0)
            m = np.isfinite(pv)
            sub = np.flatnonzero(m.ravel())
            sub = rng.choice(sub, min(sub.size, 8000), replace=False)
            rr, cc = np.unravel_index(sub, m.shape)
            xs, ys = rasterio.transform.xy(tr, rr, cc)
            for k, v in (("lat", lat), ("slope", slope), ("aspect", aspect), ("pisr", pv),
                         ("elev", dem[1:-1, 1:-1])):
                samples[k].append(v[rr, cc].astype(np.float32))
            samples["x"].append(np.asarray(xs, np.float32))
            samples["y"].append(np.asarray(ys, np.float32))
            n_done += 1
    s = {k: np.concatenate(v) for k, v in samples.items()}
    np.savez_compressed(paths.qc / "qc_samples.npz", **s)
    report["reproducibility"] = {"windows": n_done, "max_abs_diff_vs_fresh": repro_max}

    flat = s["slope"] < 0.5
    ref = reference_flat(s["lat"][flat], params)
    rel = s["pisr"][flat] / ref - 1
    by_asp = {}
    for name, lo in (("N", 337.5), ("NE", 22.5), ("E", 67.5), ("SE", 112.5), ("S", 157.5), ("SW", 202.5),
                     ("W", 247.5), ("NW", 292.5)):
        sel = (s["slope"] > 15) & (((s["aspect"] - lo) % 360) < 45)
        by_asp[name] = float(s["pisr"][sel].mean()) if sel.any() else None
    report["physics"] = {
        "n_samples": int(s["pisr"].size),
        "flat_cells": int(flat.sum()),
        "flat_rel_diff_vs_reference": {"mean": float(rel.mean()), "max_abs": float(np.abs(rel).max())},
        "mean_pisr_by_aspect_slope_gt15": by_asp,
        "ordering_S_gt_EW_gt_N": bool(by_asp["S"] > max(by_asp["E"], by_asp["W"]) > min(by_asp["E"], by_asp["W"])
                                      > by_asp["N"]),
        "E_W_rel_diff": float(abs(by_asp["E"] - by_asp["W"]) / np.mean([by_asp["E"], by_asp["W"]])),
    }
    print("physics:", json.dumps(report["physics"], indent=1), flush=True)

    # --- 6. band-boundary seams --------------------------------------------------------------------
    band_rows = 1024  # run_pisr default band height
    rows = []
    with rasterio.open(pisr_path) as P:
        for rb in range(band_rows, P.height, band_rows):
            a = P.read(1, window=Window(0, rb - 6, P.width, 12)).astype(np.float64)
            dif = np.abs(np.diff(a, axis=0))
            per_row = np.nanmean(np.where(np.isfinite(dif), dif, np.nan), axis=1) if np.isfinite(dif).any() else None
            if per_row is None or not np.isfinite(per_row).all():
                continue
            seam = per_row[5]  # difference between rows rb-1 and rb
            rows.append(dict(boundary_row=rb, seam=seam, interior_median=float(np.median(np.r_[per_row[:5],
                                                                                                    per_row[6:]]))))
    sd = pd.DataFrame(rows)
    sd["ratio"] = sd.seam / sd.interior_median
    sd.to_csv(paths.qc / "band_seams.csv", index=False)
    report["band_seams"] = {"n": len(sd), "ratio_mean": float(sd.ratio.mean()), "ratio_max": float(sd.ratio.max())}

    checks = {
        "grid": all([g["crs_ok"], g["size_ok"], g["transform_ok"], g["matches_dem"]]),
        "nodata_pattern": cnt["nan_where_expected_valid"] == 0 and cnt["valid_where_dem_nan"] == 0,
        "range": cnt["above_bound"] == 0 and cnt["below_zero"] == 0,
        "reproducible": repro_max < 1e-4,
        "flat_matches_reference": abs(report["physics"]["flat_rel_diff_vs_reference"]["mean"]) < 0.01,
        "aspect_ordering": report["physics"]["ordering_S_gt_EW_gt_N"],
        "no_band_seams": report["band_seams"]["ratio_max"] < 1.5,
    }
    report["checks"] = checks
    report["all_passed"] = all(checks.values())
    (paths.qc / "qc_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(checks, indent=1))
    print("ALL PASSED" if report["all_passed"] else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
