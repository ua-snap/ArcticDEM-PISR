"""Build the Alaska AOI, the final 3338 target grid, and the 3413 download AOI.

Outputs (under <data_root>/ancillary):
  alaska_boundary_census.gpkg  Alaska polygon from the Census cartographic boundary (EPSG:3338)
  aoi_3338_bbox.gpkg           Alaska bbox in EPSG:3338, snapped outward to the target resolution
  target_grid.json             The final product grid (crs, resolution, bounds, shape, transform)
  aoi_3413_download.gpkg       bbox densified, reprojected to EPSG:3413, and buffered
"""
import argparse
import json
import math
import zipfile

import geopandas as gpd
import requests
from shapely.geometry import box

from arcticdem_pisr.config import Paths, load_config


def fetch_boundary(url, dest_dir):
    zpath = dest_dir / url.rsplit("/", 1)[-1]
    if not zpath.exists():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as zf:
        shp = next(n for n in zf.namelist() if n.endswith(".shp"))
    return gpd.read_file(f"zip://{zpath}!{shp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    paths.mkdirs()

    tg = cfg["target_grid"]
    res = tg["resolution_m"]

    states = fetch_boundary(cfg["aoi"]["boundary_url"], paths.ancillary)
    ak = states[states["STATEFP"] == cfg["aoi"]["state_fips"]].to_crs(tg["crs"])
    assert len(ak) == 1, "expected exactly one Alaska feature"
    ak.to_file(paths.alaska_boundary, driver="GPKG")

    xmin, ymin, xmax, ymax = ak.total_bounds
    xmin = math.floor(xmin / res) * res
    ymin = math.floor(ymin / res) * res
    xmax = math.ceil(xmax / res) * res
    ymax = math.ceil(ymax / res) * res
    width, height = int((xmax - xmin) / res), int((ymax - ymin) / res)

    grid = {
        "crs": tg["crs"],
        "resolution_m": res,
        "bounds": [xmin, ymin, xmax, ymax],
        "width": width,
        "height": height,
        "transform": [res, 0.0, xmin, 0.0, -res, ymax],
        "note": "Pixel edges on multiples of resolution_m; north-up. Bounds = Alaska (Census) bbox snapped outward.",
    }
    paths.target_grid.write_text(json.dumps(grid, indent=2))

    bbox = gpd.GeoDataFrame({"name": ["alaska_bbox"]}, geometry=[box(xmin, ymin, xmax, ymax)], crs=tg["crs"])
    bbox.to_file(paths.aoi_3338, driver="GPKG")

    # Densify before reprojecting so the rectangle's edges curve correctly in polar stereographic.
    dl = bbox.copy()
    dl["geometry"] = dl.geometry.segmentize(1000)
    dl = dl.to_crs(cfg["arcticdem"]["crs"])
    dl["geometry"] = dl.geometry.buffer(cfg["aoi"]["download_buffer_m"])
    dl.to_file(paths.aoi_3413, driver="GPKG")

    print(f"Alaska bbox (EPSG:3338): {grid['bounds']}")
    print(f"Target grid: {width} x {height} px at {res} m = {width * height / 1e9:.2f} billion cells")


if __name__ == "__main__":
    main()
