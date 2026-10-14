"""Select ArcticDEM mosaic tiles that intersect the 3413 download AOI.

Tile ids come from the STAC collection (the authoritative list of tiles that exist). Tile footprints
are computed from the regular 100 km tile grid ("RR_CC"; lower-left corner at
(-4e6 + (CC-1)*1e5, -4e6 + (RR-1)*1e5) in EPSG:3413). A few footprints are checked against the
STAC items' proj:bbox.

Output: <data_root>/ancillary/tile_list_<res>m.csv (tile, row, col, xmin, ymin, xmax, ymax, item_url)
"""
import argparse
import random

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

from arcticdem_pisr.config import Paths, load_config


def tile_bounds(tile, origin, size):
    row, col = (int(v) for v in tile.split("_"))
    xmin = origin + (col - 1) * size
    ymin = origin + (row - 1) * size
    return row, col, xmin, ymin, xmin + size, ymin + size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-check", type=int, default=5, help="tiles to check against STAC proj:bbox")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    adem = cfg["arcticdem"]

    coll = requests.get(adem["stac_collection"], timeout=60).json()
    base = adem["stac_collection"].rsplit(".json", 1)[0]
    tiles = sorted(l["href"].rsplit("/", 1)[-1].removesuffix(".json")
                   for l in coll["links"] if l["rel"] == "child")
    print(f"STAC collection lists {len(tiles)} tiles")

    recs = []
    for t in tiles:
        row, col, x0, y0, x1, y1 = tile_bounds(t, adem["tile_origin_m"], adem["tile_size_m"])
        recs.append(dict(tile=t, row=row, col=col, xmin=x0, ymin=y0, xmax=x1, ymax=y1,
                         catalog_url=f"{base}/{t}.json"))
    gdf = gpd.GeoDataFrame(recs, geometry=[box(r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for r in recs],
                           crs=adem["crs"])

    aoi = gpd.read_file(paths.aoi_3413).to_crs(adem["crs"])
    sel = gdf[gdf.intersects(aoi.union_all())].copy()
    print(f"{len(sel)} tiles intersect the download AOI")

    # Check that STAC proj:bbox lies within our computed footprint plus the small overlap margin.
    # (Tiles at the edge of coverage are cropped, so the STAC bbox can be smaller than the footprint.)
    res = adem["resolution_m"]
    margin = 3 * res + 1
    for t in random.Random(0).sample(list(sel["tile"]), min(args.n_check, len(sel))):
        cat = requests.get(f"{base}/{t}.json", timeout=60).json()
        item_url = next(l["href"] for l in cat["links"] if l["rel"] == "child")
        pb = requests.get(item_url, timeout=60).json()["properties"]["proj:bbox"]
        r = sel.loc[sel.tile == t].iloc[0]
        inside = (pb[0] >= r.xmin - margin and pb[1] >= r.ymin - margin
                  and pb[2] <= r.xmax + margin and pb[3] <= r.ymax + margin)
        assert inside, f"{t}: STAC proj:bbox {pb} not within computed footprint"
        print(f"  {t}: footprint OK (STAC bbox {pb})")

    pd.DataFrame(sel.drop(columns="geometry")).to_csv(paths.tile_list, index=False)
    sel.to_file(paths.ancillary / f"tile_footprints_{res}m.gpkg", driver="GPKG")
    print(f"Wrote {paths.tile_list}")


if __name__ == "__main__":
    main()
