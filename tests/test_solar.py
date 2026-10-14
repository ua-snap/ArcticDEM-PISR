"""Tests for the PISR solar model, especially the meridian-convergence handling."""
import numpy as np
import pytest
from pyproj import CRS, Proj, Transformer
from rasterio.transform import from_origin

from arcticdem_pisr.solar import SolarParams, lat_and_jacobian, pisr

I0 = 1361.0
GEOMETRIC = SolarParams(transmittance=1.0, include_diffuse=False, step_minutes=1.0)  # extraterrestrial only
FULL = SolarParams()
RES = 32.0
N = 9  # interior size of synthetic patches


def h_horizontal(lat_deg):
    """Daily extraterrestrial irradiation on a horizontal surface at declination 0 (kWh m-2)."""
    return 24.0 / np.pi * I0 * np.cos(np.deg2rad(lat_deg)) / 1000.0


def plane_patch(crs, lon, lat, slope_deg, facing_true_az_deg):
    """DEM (with halo) of a physical plane, sampled on a grid in `crs`.

    The plane has true slope `slope_deg` and its downslope direction is `facing_true_az_deg` from TRUE north.
    Built independently of the module code: each grid cell centre is mapped to true local east/north metres
    with an azimuthal equidistant projection centred on the patch (exact distances/azimuths at the centre).
    This captures convergence, projection scale, and (for non-conformal projections like Albers) angular distortion.
    """
    crs = CRS.from_user_input(crs)
    x0, y0 = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    t = from_origin(x0 - (N / 2) * RES, y0 + (N / 2) * RES, RES, RES)
    j, i = np.meshgrid(np.arange(-1, N + 1) + 0.5, np.arange(-1, N + 1) + 0.5)
    x = t.c + j * RES
    y = t.f - i * RES
    local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +ellps=WGS84 +units=m")
    east, north = Transformer.from_crs(crs, local, always_xy=True).transform(x, y)
    face = np.deg2rad(facing_true_az_deg)
    # Elevation decreases along the facing direction.
    z = -np.tan(np.deg2rad(slope_deg)) * (east * np.sin(face) + north * np.cos(face))
    return z + 500.0, t


def centre(a):
    return float(a[N // 2, N // 2])


@pytest.mark.parametrize("lat", [52.0, 60.0, 65.0, 71.0])
def test_flat_matches_analytic(lat):
    dem, t = plane_patch("EPSG:3338", -150.0, lat, 0.0, 180.0)
    assert centre(pisr(dem, t, "EPSG:3338", GEOMETRIC)) == pytest.approx(h_horizontal(lat), rel=1e-4)


# South-facing slope beta at latitude phi is parallel to a horizontal surface at phi - beta (declination 0).
@pytest.mark.parametrize("crs", ["EPSG:3338", "EPSG:3413", "EPSG:32606"])
@pytest.mark.parametrize("lon", [-179.5, -165.0, -150.0, -135.0, 175.0])
@pytest.mark.parametrize("slope", [10.0, 30.0])
def test_south_slope_with_convergence(crs, lon, slope):
    if crs == "EPSG:32606" and abs(lon + 150.0) > 10:
        pytest.skip("UTM 6N is only meaningful near its zone")
    lat = 62.0
    dem, t = plane_patch(crs, lon, lat, slope, 180.0)
    got = centre(pisr(dem, t, crs, GEOMETRIC))
    assert got == pytest.approx(h_horizontal(lat - slope), rel=2e-3)


@pytest.mark.parametrize("crs", ["EPSG:3338", "EPSG:3413"])
@pytest.mark.parametrize("lon", [-170.0, -150.0, -132.0, 178.0])
def test_east_west_symmetry_and_ordering(crs, lon):
    lat, slope = 64.0, 25.0
    v = {}
    for name, az in [("N", 0.0), ("E", 90.0), ("S", 180.0), ("W", 270.0)]:
        dem, t = plane_patch(crs, lon, lat, slope, az)
        v[name] = centre(pisr(dem, t, crs, FULL))
    flat_dem, t = plane_patch(crs, lon, lat, 0.0, 180.0)
    flat = centre(pisr(flat_dem, t, crs, FULL))
    assert v["E"] == pytest.approx(v["W"], rel=1e-3)
    assert v["S"] > flat > v["N"]
    assert v["S"] > v["E"] > v["N"]


@pytest.mark.parametrize("lon", [-175.0, -150.0, -130.0, 172.0])
def test_grid_invariance_3338_vs_3413(lon):
    """The same physical slope gives the same PISR whichever projection the grid is in."""
    lat = 61.0
    a = centre(pisr(*plane_patch("EPSG:3338", lon, lat, 20.0, 135.0), "EPSG:3338", FULL))
    b = centre(pisr(*plane_patch("EPSG:3413", lon, lat, 20.0, 135.0), "EPSG:3413", FULL))
    assert a == pytest.approx(b, rel=2e-3)


@pytest.mark.parametrize("lon,lat", [(-150.0, 62.0), (-131.0, 56.0), (174.0, 52.5), (-157.0, 71.3)])
@pytest.mark.parametrize("crs", ["EPSG:3338", "EPSG:3413"])
def test_jacobian_matches_pyproj_factors(crs, lon, lat):
    """Jacobian columns give PROJ's meridional/parallel scales, convergence, and meridian-parallel angle."""
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    t = from_origin(x - RES / 2, y + RES / 2, RES, RES)
    _, jxe, jye, jxn, jyn = (float(v[0, 0]) for v in lat_and_jacobian(t, (1, 1), crs))
    f = Proj(crs).get_factors(lon, lat)
    assert np.hypot(jxn, jyn) == pytest.approx(f.meridional_scale, rel=1e-5)
    assert np.hypot(jxe, jye) == pytest.approx(f.parallel_scale, rel=1e-5)
    theta = np.degrees(np.arctan2(jxn, jyn))  # grid azimuth of true north
    assert theta == pytest.approx(-f.meridian_convergence, abs=1e-3)
    angle = np.degrees(np.arccos((jxe * jxn + jye * jyn) / np.hypot(jxe, jye) / np.hypot(jxn, jyn)))
    assert angle == pytest.approx(f.meridian_parallel_angle, abs=1e-3)


def test_nodata_propagates():
    dem, t = plane_patch("EPSG:3338", -150.0, 62.0, 10.0, 180.0)
    dem[3, 3] = np.nan
    out = pisr(dem, t, "EPSG:3338", FULL)
    assert np.isnan(out[1:4, 1:4]).all()  # interior cells whose 3x3 window touches the NaN
    assert np.isfinite(out[6:, 6:]).all()


def test_full_model_reasonable_magnitudes():
    dem, t = plane_patch("EPSG:3338", -150.0, 62.0, 0.0, 180.0)
    flat = centre(pisr(dem, t, "EPSG:3338", FULL))
    assert 0.3 * h_horizontal(62.0) < flat < 0.8 * h_horizontal(62.0)


def test_reference_flat_matches_plane():
    from arcticdem_pisr.solar import reference_flat
    for lat in (55.0, 65.0):
        dem, t = plane_patch("EPSG:3338", -150.0, lat, 0.0, 180.0)
        assert reference_flat([lat], FULL)[0] == pytest.approx(centre(pisr(dem, t, "EPSG:3338", FULL)), rel=1e-6)
