"""Package the PISR GeoTIFF as a CF-1.8 netCDF4 file for building a Rasdaman coverage (Phase 3a).

- dims (y, x); 1-D cell-centre coordinates in metres; y descending (north-up, same as the GeoTIFF)
- `pisr` float32, NaN fill, grid_mapping "crs"; `crs` variable with CF Albers parameters + WKT (EPSG:3338)
- `elevation` float32: the void-filled 32 m EPSG:3338 DEM that PISR was computed from
- zlib + shuffle, chunked; provenance in global attributes
Then verifies: GDAL reads CRS/geotransform identically to the GeoTIFF, and values round-trip at random points.
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
from osgeo import gdal
from pyproj import CRS

from arcticdem_pisr.config import REPO_ROOT, Paths, load_config

gdal.UseExceptions()


def git_version():
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--complevel", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    paths = Paths(cfg)
    res, ver = cfg["arcticdem"]["resolution_m"], cfg["arcticdem"]["version"]
    src = Path(args.src or paths.pisr / f"pisr_equinox_{res}m_3338.tif")
    out = Path(args.out or paths.output / f"arcticdem_v{ver}_pisr_equinox_{res}m_3338.nc")
    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src) as ds:
        tags = ds.tags()
        crs = CRS.from_user_input(ds.crs)
    da = xr.open_dataarray(src, engine="rasterio", chunks={"x": 4 * args.chunk, "y": 4 * args.chunk})
    da = da.squeeze("band", drop=True).astype("float32")
    da = da.drop_vars("spatial_ref", errors="ignore")

    da.name = "pisr"
    da.attrs = {
        "long_name": "Potential incoming solar radiation, clear-sky total (direct + diffuse), equinox daily total",
        "units": "kWh m-2 day-1",
        "grid_mapping": "crs",
        "comment": ("Idealized equinox (solar declination 0, mean Earth-Sun distance); local slope/aspect "
                    "self-shading only, no cast shadows from surrounding terrain; constant lumped atmospheric "
                    "transmittance (no elevation effect); ground-reflected radiation excluded. Water is NoData: cells "
                    "ArcticDEM flags as no stereo data (datamask 0) that are flattened surfaces (slope < %s deg)."
                    % cfg["water_mask"]["max_slope_deg"]),
    }
    x = da["x"].values
    y = da["y"].values
    dem = xr.open_dataarray(paths.dem_3338_filled, engine="rasterio", chunks={"x": 4 * args.chunk, "y": 4 * args.chunk})
    dem = dem.squeeze("band", drop=True).astype("float32").drop_vars("spatial_ref", errors="ignore")
    assert np.array_equal(dem.x, da.x) and np.array_equal(dem.y, da.y), "DEM and PISR grids differ"
    dem.name = "elevation"
    dem.attrs = {
        "long_name": "Surface elevation used for PISR (ArcticDEM v4.1 mosaic, warped to this grid)",
        "standard_name": "height_above_reference_ellipsoid",
        "units": "m",
        "grid_mapping": "crs",
        "comment": ("Heights above the WGS84 ellipsoid (ArcticDEM convention). Cubic resampling from 32 m "
                    "EPSG:3413. Enclosed voids of <= %s cells were filled by inverse-distance interpolation."
                    % cfg["void_fill"]["max_cells"]),
    }
    tii = xr.open_dataarray(paths.tii_3338, engine="rasterio", chunks={"x": 4 * args.chunk, "y": 4 * args.chunk})
    tii = tii.squeeze("band", drop=True).astype("float32").drop_vars("spatial_ref", errors="ignore")
    assert np.array_equal(tii.x, da.x) and np.array_equal(tii.y, da.y), "TII and PISR grids differ"
    ref_lat = cfg["tii"]["ref_lat_deg"]
    tii.name = "terrain_insolation_index"
    tii.attrs = {
        "long_name": "Terrain insolation index: equinox insolation relative to flat ground, latitude removed",
        "units": "kWh m-2 day-1",
        "grid_mapping": "crs",
        "reference_latitude_deg": float(ref_lat),
        "comment": (f"Same model as pisr, but every cell is evaluated at latitude {ref_lat}N (keeping its true slope "
                    f"and aspect) and the flat-ground value at {ref_lat}N is subtracted. 0 = like flat ground, "
                    "> 0 = warm (sun-facing) slope, < 0 = cold slope. The same slope and aspect give the same "
                    "value anywhere, so the index carries terrain only (latitude and elevation removed). "
                    "Water is NoData."),
    }
    ds_out = xr.Dataset({"pisr": da, "terrain_insolation_index": tii, "elevation": dem})
    ds_out["x"].attrs = {"standard_name": "projection_x_coordinate", "long_name": "x coordinate of cell centre",
                         "units": "m", "axis": "X"}
    ds_out["y"].attrs = {"standard_name": "projection_y_coordinate", "long_name": "y coordinate of cell centre",
                         "units": "m", "axis": "Y"}
    cf = crs.to_cf()
    cf["crs_wkt"] = crs.to_wkt()
    cf["spatial_ref"] = crs.to_wkt()  # read by GDAL
    cf["epsg_code"] = "EPSG:3338"
    cf["GeoTransform"] = " ".join(str(v) for v in [x[0] - res / 2, res, 0, y[0] + res / 2, 0, -res])
    ds_out["crs"] = xr.DataArray(np.int32(0), attrs=cf)

    model = {k: (int(v) if k == "pisr_include_diffuse" else float(v)) for k, v in tags.items() if k.startswith("pisr_")}
    ds_out.attrs = {
        "Conventions": "CF-1.8",
        "title": f"Potential incoming solar radiation (equinox, clear sky) for Alaska from ArcticDEM v{ver} {res} m",
        "summary": ("Daily clear-sky total insolation on the terrain surface for an idealized equinox day, computed "
                    "from the ArcticDEM mosaic (warped from EPSG:3413 to EPSG:3338 with cubic resampling; small "
                    "enclosed voids filled by inverse-distance interpolation). "
                    "pisr is a combined slope/aspect/latitude covariate; terrain_insolation_index is the "
                    "latitude-independent slope/aspect effect (warm vs cold slopes). Extent: Alaska's bounding box in "
                    "EPSG:3338, including adjacent Canada and Russia where ArcticDEM has data."),
        "source": (f"ArcticDEM mosaic v{ver}, {res} m (Polar Geospatial Center, University of Minnesota), "
                   "https://www.pgc.umn.edu/data/arcticdem/"),
        # Wording per https://www.pgc.umn.edu/guides/user-services/acknowledgement-policy/ (checked 2026-10-14)
        "acknowledgment": ("DEMs provided by the Polar Geospatial Center under NSF-OPP awards 1043681, 1559691, "
                           "1542736, 1810976, and 2129685."),
        "references": ("Porter, C., Howat, I., Noh, M.J., Husby, E., Khuvis, S., Danish, E., Tomko, K., Gardiner, J., "
                       "Negrete, A., Yadav, B., Klassen, J., Kelleher, C., Cloutier, M., Bakker, J., Enos, J., "
                       "Arnold, G., Bauer, G., and Morin, P., 2023, \"ArcticDEM - Mosaics, Version 4.1\", "
                       "https://doi.org/10.7910/DVN/3VDC4W, Harvard Dataverse, V1, accessed 2026-10-14."),
        "method": ("Horn (1981) gradients converted to true east/north via the local projection Jacobian; sun "
                   "geometry at declination 0 in hour-angle steps; direct = I0*tau^m*cos(i) with Kasten & Young "
                   "(1989) air mass; diffuse = I0*sin(h)*(0.271-0.294*tau^m)*(1+cos(slope))/2 (Liu & Jordan 1960)."),
        **model,
        "software": f"https://github.com/ua-snap/ArcticDEM-PISR ({git_version()})",
        "history": f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}: created from {src.name}",
        "date_created": datetime.now(timezone.utc).date().isoformat(),
    }
    var_enc = {"zlib": True, "complevel": args.complevel, "shuffle": True, "_FillValue": np.float32(np.nan),
               "chunksizes": (args.chunk, args.chunk), "dtype": "float32"}
    enc = {"pisr": var_enc, "terrain_insolation_index": dict(var_enc), "elevation": dict(var_enc),
           "x": {"_FillValue": None}, "y": {"_FillValue": None}}
    tmp = out.with_suffix(".tmp.nc")
    ds_out.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=enc)
    tmp.rename(out)
    print(f"Wrote {out}")

    # --- verification ---
    g_nc = gdal.Open(f'NETCDF:"{out}":pisr')
    g_tif = gdal.Open(str(src))
    assert np.allclose(g_nc.GetGeoTransform(), g_tif.GetGeoTransform()), (g_nc.GetGeoTransform(),
                                                                         g_tif.GetGeoTransform())
    assert CRS.from_wkt(g_nc.GetProjection()).equals(crs), "CRS mismatch"
    assert (g_nc.RasterXSize, g_nc.RasterYSize) == (g_tif.RasterXSize, g_tif.RasterYSize)
    rng = np.random.default_rng(0)
    nc = xr.open_dataset(out)
    with rasterio.open(src) as ds:
        for _ in range(200):
            r, c = int(rng.integers(ds.height)), int(rng.integers(ds.width))
            a = ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]
            b = nc["pisr"].isel(y=r, x=c).item()
            assert (np.isnan(a) and np.isnan(b)) or a == b, (r, c, a, b)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    (out.parent / "SHA256SUMS").write_text(f"{h.hexdigest()}  {out.name}\n")
    print(json.dumps({"verified": True, "gdal_geotransform": g_nc.GetGeoTransform(), "sha256": h.hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
