"""Maps and plots for sharing the PISR product (Phase 3c). Writes PNGs to <data_root>/figures/.

Uses outputs of qc.py (qc_samples.npz, pisr_histogram.csv). Palette: single-hue orange sequential ramp for
PISR magnitude; blue ordinal ramp for slope classes; categorical slots 1/2 for two-series charts
(validated with the dataviz skill's validator).
"""
import argparse
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Transformer
from rasterio.windows import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arcticdem_pisr.config import Paths, load_config  # noqa: E402
from arcticdem_pisr.solar import SolarParams, reference_flat  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NODATA = "#dcdcd6"
SEQ = LinearSegmentedColormap.from_list("pisr", ["#f6cdb6", "#f19a72", "#eb6834", "#c94e1f", "#9c3a14", "#6b270c"])
SEQ.set_bad(NODATA)
SLOT1, SLOT2 = "#2a78d6", "#eb6834"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
UNITS = "kWh m$^{-2}$ day$^{-1}$"
REGIONS = [  # (label, lon, lat)
    ("Alaska Range (Denali)", -151.0, 63.07),
    ("Brooks Range", -150.6, 68.1),
    ("Juneau Icefield", -134.3, 58.6),
    ("Kiska Island, Aleutians (W of 180°)", 177.55, 52.0),
]
HALF = 30000  # zoom half-width, m

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def hillshade(z, res, az=315, alt=45):
    gy, gx = np.gradient(z, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, h = np.deg2rad(az), np.deg2rad(alt)
    return np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect)


def clean_map_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def statewide(paths, pisr_path, ak, vmin, vmax):
    with rasterio.open(pisr_path) as ds:
        dec = 24
        arr = ds.read(1, out_shape=(ds.height // dec, ds.width // dec), masked=True)
        b = ds.bounds
    fig, ax = plt.subplots(figsize=(13, 7.6))
    im = ax.imshow(arr, cmap=SEQ, vmin=vmin, vmax=vmax, extent=[b.left, b.right, b.bottom, b.top],
                   interpolation="nearest")
    ak.boundary.plot(ax=ax, color=INK2, lw=0.35)
    to = Transformer.from_crs(4326, 3338, always_xy=True)
    for i, (label, lon, lat) in enumerate(REGIONS, 1):
        x, y = to.transform(lon, lat)
        ax.add_patch(plt.Rectangle((x - HALF, y - HALF), 2 * HALF, 2 * HALF, fill=False, ec=INK, lw=1.0))
        ax.annotate(f"{i}", (x + HALF, y + HALF), xytext=(3, 3), textcoords="offset points", color=INK,
                    fontsize=10, fontweight="bold")
    clean_map_axes(ax)
    ax.set_title("Potential incoming solar radiation at the equinox (clear sky, daily total), 32 m")
    ax.text(0, 1.0, "Alaska bounding box, EPSG:3338 · ArcticDEM v4.1 · gray = no DEM data · numbered boxes = zoom panels",
            transform=ax.transAxes, color=INK2, va="bottom", fontsize=9)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50, shrink=0.6)
    cb.set_label(UNITS, color=INK2); cb.outline.set_visible(False)
    out = paths.figures / "fig1_statewide_pisr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def zooms(paths, pisr_path, dem_path, vmin, vmax):
    to = Transformer.from_crs(4326, 3338, always_xy=True)
    fig, axes = plt.subplots(len(REGIONS), 2, figsize=(10, 4.9 * len(REGIONS)), constrained_layout=True)
    with rasterio.open(pisr_path) as P, rasterio.open(dem_path) as D:
        for i, (label, lon, lat) in enumerate(REGIONS):
            x, y = to.transform(lon, lat)
            bnd = (x - HALF, y - HALF, x + HALF, y + HALF)
            p = P.read(1, window=from_bounds(*bnd, transform=P.transform), masked=True)
            z = D.read(1, window=from_bounds(*bnd, transform=D.transform)).astype(float)
            hs = hillshade(np.where(np.isfinite(z), z, np.nanmean(z)), P.res[0])
            hs = np.ma.masked_where(~np.isfinite(z), hs)
            cm = plt.get_cmap("gray").copy(); cm.set_bad(NODATA)
            axes[i, 0].imshow(hs, cmap=cm, vmin=-0.2, vmax=1.0, interpolation="lanczos")
            im = axes[i, 1].imshow(p, cmap=SEQ, vmin=vmin, vmax=vmax, interpolation="lanczos")
            axes[i, 0].set_title(f"{i + 1}. {label}: hillshade")
            axes[i, 1].set_title(f"{i + 1}. PISR")
            for a in axes[i]:
                clean_map_axes(a)
            fig.colorbar(im, ax=axes[i, 1], shrink=0.75, label=UNITS).outline.set_visible(False)
    fig.suptitle("Zoom panels (60 km × 60 km, full 32 m resolution; north is up in EPSG:3338)", x=0.01, ha="left",
                 color=INK2, fontsize=10)
    out = paths.figures / "fig2_zoom_panels.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def distribution(paths):
    h = pd.read_csv(paths.qc / "pisr_histogram.csv")
    mid = (h.bin_lo + h.bin_hi) / 2
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for col, color, name in (("count_alaska", SLOT1, "Alaska (Census boundary)"),
                             ("count_bbox", SLOT2, "Whole bounding box (incl. Canada, Russia)")):
        dens = h[col] / h[col].sum() / (h.bin_hi - h.bin_lo)
        ax.plot(mid, dens, color=color, lw=1.8, label=name)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, None)
    ax.yaxis.grid(True)
    ax.set_xlabel(f"PISR ({UNITS})"); ax.set_ylabel("Density")
    ax.set_title("Distribution of PISR values")
    ax.legend(loc="upper right")
    out = paths.figures / "fig3_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def latitude(paths, s, params):
    flat = s["slope"] < 0.5
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(flat)
    idx = rng.choice(idx, min(idx.size, 20000), replace=False)
    lats = np.linspace(np.floor(s["lat"].min()), np.ceil(s["lat"].max()), 200)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.scatter(s["lat"][idx], s["pisr"][idx], s=8, color=SLOT1, alpha=0.35, lw=0, label="Sampled flat cells (slope < 0.5°)")
    ax.plot(lats, reference_flat(lats, params), color=SLOT2, lw=2, label="Model reference, horizontal surface")
    ax.yaxis.grid(True)
    ax.set_xlabel("Latitude (°N)"); ax.set_ylabel(f"PISR ({UNITS})")
    ax.set_title("Flat terrain follows the expected latitude curve")
    ax.legend(loc="upper right")
    out = paths.figures / "fig4_flat_vs_latitude.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def aspect(paths, s):
    edges = np.arange(0, 361, 15)
    mids = edges[:-1] + 7.5
    classes = [(0, 5), (5, 15), (15, 30), (30, 90)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (lo, hi), color in zip(classes, ORDINAL):
        m = (s["slope"] >= lo) & (s["slope"] < hi) & (s["lat"] > 60) & (s["lat"] < 66)
        idx = np.digitize(s["aspect"][m], edges) - 1
        means = np.array([s["pisr"][m][idx == k].mean() if (idx == k).sum() > 50 else np.nan for k in range(len(mids))])
        label = f"{lo}–{hi}°" if hi < 90 else f">{lo}°"
        ax.plot(mids, means, color=color, lw=2, marker="o", ms=4, label=f"Slope {label}")
        ax.annotate(label, (mids[-1], means[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360], ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    ax.set_xlim(0, 375)
    ax.yaxis.grid(True)
    ax.set_xlabel("Aspect (direction the slope faces, true north)"); ax.set_ylabel(f"Mean PISR ({UNITS})")
    ax.set_title("PISR as a slope/aspect proxy: south-facing steep slopes highest (cells at 60–66°N)")
    ax.legend(loc="upper left")
    out = paths.figures / "fig5_aspect_slope.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    paths.figures.mkdir(exist_ok=True)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
    params = SolarParams(**cfg.get("pisr", {}))
    pisr_path = paths.pisr / f"pisr_equinox_{res}m_3338.tif"
    dem_path = paths.dem / f"arcticdem_v{ver}_{res}m_dem_3338.tif"
    ak = gpd.read_file(paths.alaska_boundary).to_crs(3338)
    s = dict(np.load(paths.qc / "qc_samples.npz"))
    vmin, vmax = np.percentile(s["pisr"], [1, 99])
    vmin, vmax = np.floor(vmin * 2) / 2, np.ceil(vmax * 2) / 2
    outs = [statewide(paths, pisr_path, ak, vmin, vmax), zooms(paths, pisr_path, dem_path, vmin, vmax),
            distribution(paths), latitude(paths, s, params), aspect(paths, s)]
    print(json.dumps([str(o) for o in outs], indent=1))


if __name__ == "__main__":
    main()
