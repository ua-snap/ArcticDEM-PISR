"""Maps and plots for sharing the PISR / terrain insolation index product (Phase 3c). PNGs -> <data_root>/figures/.

Uses outputs of qc.py (qc_samples.npz, pisr_histogram.csv, tii_histogram.csv).
  fig1_pisr_map.png          full domain + 3 sample areas (Brooks Range, Fairbanks, Denali), sequential orange
  fig2_tii_explained.png     why TII: PISR vs latitude for flat / south / north slopes, and TII for the same surfaces
  fig3_tii_map.png           full domain + 3 sample areas, diverging blue (cold) - white (0) - red (warm)
  fig4_tii_aspect_slope.png  TII vs aspect by slope class, all latitudes
  fig5_distributions.png     PISR and TII distributions over Alaska
  fig6_flat_vs_latitude.png  QC: flat cells follow the model's latitude curve
Palettes: single-hue orange ramp for PISR magnitude; blue-white-red diverging for TII (white = flat ground, as the
user requested); blue ordinal ramp for slope classes (validated with the dataviz skill's validator).
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
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arcticdem_pisr.config import Paths, load_config  # noqa: E402
from arcticdem_pisr.solar import SolarParams, pisr, reference_flat  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NODATA = "#dcdcd6"
SEQ = LinearSegmentedColormap.from_list("pisr", ["#f6cdb6", "#f19a72", "#eb6834", "#c94e1f", "#9c3a14", "#6b270c"])
SEQ.set_bad(NODATA)
# PISR maps use the reversed ramp (more sun = brighter). With dark = high, sunny south faces looked dark and the
# terrain read as if lit from the north.
SUN = SEQ.reversed()
SUN.set_bad(NODATA)
DIV = LinearSegmentedColormap.from_list("coldwarm", ["#0d366b", "#2a78d6", "#9ec5f4", "#ffffff", "#f4a6a5", "#e34948",
                                                     "#7a1d1c"])
DIV.set_bad(NODATA)
WARM, COLD, NEUTRAL = "#e34948", "#2a78d6", "#52514e"
SLOT1 = "#2a78d6"
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
UNITS = "kWh m$^{-2}$ day$^{-1}$"
REGIONS = [  # (label, lon, lat) - sample areas, north to south
    ("Brooks Range", -150.6, 68.1),
    ("Fairbanks", -147.72, 64.84),
    ("Denali", -151.0, 63.07),
]
HALF = 30000  # sample-area half-width, m (60 km boxes)
TII_LIM = 1.5

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def clean_map_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


def domain_with_panels(raster, ak, cmap, vmin, vmax, title, subtitle, ticks, ticklabels, out, panel_note=None):
    """Full-domain map on top (sample-area boxes labelled) and the three 60 km sample areas below."""
    to = Transformer.from_crs(4326, 3338, always_xy=True)
    fig = plt.figure(figsize=(14, 12.6))
    ax0 = fig.add_axes([0.02, 0.42, 0.96, 0.52])
    with rasterio.open(raster) as ds:
        dec = 24
        a = ds.read(1, out_shape=(ds.height // dec, ds.width // dec), masked=True)
        b = ds.bounds
    im = ax0.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, extent=[b.left, b.right, b.bottom, b.top],
                    interpolation="nearest")
    ak.boundary.plot(ax=ax0, color=INK2, lw=0.35)
    for label, lon, lat in REGIONS:
        x, y = to.transform(lon, lat)
        ax0.add_patch(plt.Rectangle((x - HALF, y - HALF), 2 * HALF, 2 * HALF, fill=False, ec=INK, lw=1.3))
        ax0.annotate(label, (x + HALF, y), xytext=(5, 0), textcoords="offset points", va="center", color=INK,
                     fontsize=10, fontweight="bold",
                     bbox=dict(fc=SURFACE, ec="none", alpha=0.75, pad=1.5))
    clean_map_axes(ax0)
    ax0.set_title(title, pad=22)
    ax0.text(0, 1.01, subtitle, transform=ax0.transAxes, color=INK2, va="bottom", fontsize=9)
    cb = fig.colorbar(im, ax=ax0, orientation="horizontal", fraction=0.035, pad=0.01, aspect=50, shrink=0.55)
    cb.set_ticks(ticks); cb.set_ticklabels(ticklabels)
    cb.set_label(UNITS, color=INK2, labelpad=4); cb.outline.set_visible(False)
    cb.ax.xaxis.set_label_position("top")
    for k, (label, lon, lat) in enumerate(REGIONS):
        ax = fig.add_axes([0.02 + k * 0.33, 0.015, 0.31, 0.355])
        x, y = to.transform(lon, lat)
        with rasterio.open(raster) as ds:
            z = ds.read(1, window=from_bounds(x - HALF, y - HALF, x + HALF, y + HALF, ds.transform), masked=True)
        ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos")
        clean_map_axes(ax)
        ax.set_title(f"{label} ({lat:.1f}°N), 60 × 60 km", fontsize=10)
        if panel_note:
            ax.text(0.01, 0.01, panel_note(z), transform=ax.transAxes, fontsize=9, color=INK,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plane_pisr(slope_deg, aspect_deg, lat_deg, params):
    """PISR of an ideal plane (true slope/aspect) at a latitude, via the production kernel (identity projection)."""
    t = from_origin(0, 3, 1, 1)
    j, i = np.meshgrid(np.arange(3) + 0.5, np.arange(3) + 0.5)
    east, north = j, -i
    face = np.deg2rad(aspect_deg)
    z = -np.tan(np.deg2rad(slope_deg)) * (east * np.sin(face) + north * np.cos(face))
    one, zero = np.ones((1, 1)), np.zeros((1, 1))
    geo = (np.full((1, 1), np.deg2rad(lat_deg)), one, zero, zero, one)
    return float(pisr(z, from_origin(1, 2, 1, 1), "EPSG:3338", params, geo=geo)[0, 0])


def tii_explained(paths, params, ref_lat, out):
    lats = np.arange(52, 71.01, 0.5)
    surfaces = [("30° south-facing slope", 30, 180, WARM), ("flat ground", 0, 180, NEUTRAL),
                ("30° north-facing slope", 30, 0, COLD)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    flat_ref = reference_flat([ref_lat], params)[0]
    for label, sl, asp, color in surfaces:
        p = np.array([plane_pisr(sl, asp, la, params) for la in lats])
        tii = plane_pisr(sl, asp, ref_lat, params) - flat_ref
        axes[0].plot(lats, p, color=color, lw=2.2)
        axes[0].annotate(label, (lats[-1], p[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                         color=INK2, fontsize=9)
        axes[1].plot(lats, np.full_like(lats, tii), color=color, lw=2.2)
        axes[1].annotate(f"{label}: {tii:+.2f}", (lats[-1], tii), xytext=(6, 0), textcoords="offset points",
                         va="center", color=INK2, fontsize=9)
    axes[0].set_title("PISR: latitude and terrain mixed together")
    axes[0].set_ylabel(f"PISR ({UNITS})")
    axes[1].set_title(f"Terrain insolation index: terrain only")
    axes[1].set_ylabel(f"TII ({UNITS})")
    axes[1].axhline(0, color=AXIS, lw=0.8)
    axes[1].set_ylim(-TII_LIM - 0.3, TII_LIM + 0.3)
    for ax in axes:
        ax.set_xlabel("Latitude (°N)")
        ax.set_xlim(52, 77)
        ax.set_xticks([52, 56, 60, 64, 68, 71])
        ax.yaxis.grid(True)
    fig.text(0.5, -0.04, f"TII = PISR computed as if the cell were at {ref_lat:.0f}°N (true slope and aspect kept) − "
                         f"PISR of flat ground at {ref_lat:.0f}°N. Same slope + aspect → same TII anywhere.",
             ha="center", color=INK2, fontsize=10)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def tii_aspect(s, out):
    edges = np.arange(0, 361, 15)
    mids = edges[:-1] + 7.5
    classes = [(0, 5), (5, 15), (15, 30), (30, 90)]
    t = s["tii"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for (lo, hi), color in zip(classes, ORDINAL):
        m = (s["slope"] >= lo) & (s["slope"] < hi) & np.isfinite(t)
        idx = np.digitize(s["aspect"][m], edges) - 1
        means = np.array([t[m][idx == k].mean() if (idx == k).sum() > 50 else np.nan for k in range(len(mids))])
        label = f"{lo}–{hi}°" if hi < 90 else f">{lo}°"
        ax.plot(mids, means, color=color, lw=2, marker="o", ms=4, label=f"Slope {label}")
        ax.annotate(label, (mids[-1], means[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360], ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    ax.set_xlim(0, 375)
    ax.yaxis.grid(True)
    ax.set_xlabel("Aspect (direction the slope faces, true north)")
    ax.set_ylabel(f"Mean TII ({UNITS})")
    ax.set_title("Terrain insolation index by aspect and slope (all of Alaska)")
    ax.legend(loc="upper left")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def distributions(paths, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for ax, csv, name, xlim in ((axes[0], "pisr_histogram.csv", "PISR", (0, 6)),
                                (axes[1], "tii_histogram.csv", "Terrain insolation index", (-3, 3))):
        h = pd.read_csv(paths.qc / csv)
        mid = (h.bin_lo + h.bin_hi) / 2
        dens = h.count_alaska / h.count_alaska.sum() / (h.bin_hi - h.bin_lo)
        ax.plot(mid, dens, color=SLOT1, lw=1.8)
        ax.set_xlim(*xlim); ax.set_ylim(0, None)
        ax.yaxis.grid(True)
        ax.set_xlabel(f"{name} ({UNITS})"); ax.set_ylabel("Density")
        ax.set_title(f"{name}, Alaska")
    axes[1].axvline(0, color=AXIS, lw=0.8)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def latitude(s, params, out):
    flat = s["slope"] < 0.5
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(flat)
    idx = rng.choice(idx, min(idx.size, 20000), replace=False)
    lats = np.linspace(np.floor(s["lat"].min()), np.ceil(s["lat"].max()), 200)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.scatter(s["lat"][idx], s["pisr"][idx], s=8, color=SLOT1, alpha=0.35, lw=0,
               label="Sampled flat cells (slope < 0.5°)")
    ax.plot(lats, reference_flat(lats, params), color="#eb6834", lw=2, label="Model reference, horizontal surface")
    ax.yaxis.grid(True)
    ax.set_xlabel("Latitude (°N)"); ax.set_ylabel(f"PISR ({UNITS})")
    ax.set_title("QC: flat terrain follows the expected latitude curve")
    ax.legend(loc="upper right")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    paths.figures.mkdir(exist_ok=True)
    params = SolarParams(**cfg.get("pisr", {}))
    ref_lat = cfg["tii"]["ref_lat_deg"]
    ak = gpd.read_file(paths.alaska_boundary).to_crs(3338)
    s = dict(np.load(paths.qc / "qc_samples.npz"))
    F = paths.figures

    def warm_cold(z):
        z = z.compressed()
        return f"{100 * (z > 0.05).mean():.0f}% warm · {100 * (z < -0.05).mean():.0f}% cold"

    outs = [
        domain_with_panels(paths.pisr_3338, ak, SUN, 0.5, 4.5,
                           "Potential incoming solar radiation (PISR): equinox, clear sky, daily total, 32 m",
                           "Brighter = more sun · includes latitude (less sun farther north) · gray = no data or water"
                           " · boxes = sample areas",
                           [0.5, 1.5, 2.5, 3.5, 4.5], ["0.5", "1.5", "2.5", "3.5", "4.5"],
                           F / "fig1_pisr_map.png"),
        tii_explained(paths, params, ref_lat, F / "fig2_tii_explained.png"),
        domain_with_panels(paths.tii_3338, ak, DIV, -TII_LIM, TII_LIM,
                           "Terrain insolation index (TII): warm vs cold slopes, latitude removed, 32 m",
                           "White = like flat ground · red = sun-facing (warm) · blue = shaded (cold) · "
                           "gray = no data or water",
                           [-1.5, -0.75, 0, 0.75, 1.5],
                           ["−1.5\ncold slope", "−0.75", "0\nlike flat ground", "+0.75", "+1.5\nwarm slope"],
                           F / "fig3_tii_map.png", panel_note=warm_cold),
        tii_aspect(s, F / "fig4_tii_aspect_slope.png"),
        distributions(paths, F / "fig5_distributions.png"),
        latitude(s, params, F / "fig6_flat_vs_latitude.png"),
    ]
    print(json.dumps([str(o) for o in outs], indent=1))


if __name__ == "__main__":
    main()
