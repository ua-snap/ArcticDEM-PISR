"""Independent cross-checks of the PISR model against pvlib.

1. Exact reference (tight tolerance): pvlib's analytical sun position with declination fixed at 0, plus
   pvlib's angle-of-incidence, air-mass, and isotropic-sky functions. Same physics as our model, but
   different code (spherical formulas and azimuth angles, versus our vector form), on planes built on
   3338/3413 grids. This exercises the convergence handling.
2. Real-date realism check (loose tolerance): pvlib SPA sun positions over the 24 h centred on each real
   2025 equinox instant, averaged. Real declination drifts ±0.2° over a window, which shifts daily totals by
   up to ~1% at 70°N (sensitivity ∝ tan φ), so only ~1.5% agreement is expected.
"""
import numpy as np
import pandas as pd
import pytest

pvlib = pytest.importorskip("pvlib")

from arcticdem_pisr.solar import SolarParams, pisr  # noqa: E402
from test_solar import centre, plane_patch  # noqa: E402

P = SolarParams()
EQUINOXES_UTC = [pd.Timestamp("2025-03-20 09:01", tz="UTC"), pd.Timestamp("2025-09-22 18:19", tz="UTC")]
SITES = [(52.0, 177.0), (55.5, -132.0), (61.2, -149.9), (63.1, -151.0), (68.0, -150.0), (71.3, -156.8)]
SURFACES = [(0, 180), (15, 180), (35, 180), (25, 90), (25, 270), (30, 0), (20, 225)]


def poa_daily(zen_deg, az_deg, tilt, surface_az, dt_hours):
    """Daily POA total (kWh m-2) from sun positions, using pvlib irradiance/atmosphere functions."""
    up = zen_deg < 90
    am = pvlib.atmosphere.get_relative_airmass(np.where(up, zen_deg, np.nan), model="kastenyoung1989")
    tb = np.where(up, P.transmittance ** np.nan_to_num(am, nan=0.0), 0.0)
    aoi = pvlib.irradiance.aoi(tilt, surface_az, zen_deg, az_deg)
    direct = np.where(up, P.solar_constant * tb * np.clip(np.cos(np.deg2rad(aoi)), 0, None), 0.0)
    dhi = np.where(up, P.solar_constant * np.cos(np.deg2rad(zen_deg)) * np.clip(0.271 - 0.294 * tb, 0, None), 0.0)
    diffuse = pvlib.irradiance.isotropic(tilt, dhi)
    return float(np.sum(direct + diffuse) * dt_hours / 1000.0)


def pvlib_exact(lat, tilt, surface_az):
    n = 1440
    w = np.deg2rad(-180 + (np.arange(n) + 0.5) * 360 / n)
    z = pvlib.solarposition.solar_zenith_analytical(np.deg2rad(lat), w, 0.0)
    a = pvlib.solarposition.solar_azimuth_analytical(np.deg2rad(lat), w, 0.0, z)
    return poa_daily(np.degrees(z), np.degrees(a), tilt, surface_az, 24.0 / n)


def pvlib_real(lat, lon, tilt, surface_az):
    vals = []
    for eq in EQUINOXES_UTC:
        times = pd.date_range(eq - pd.Timedelta(hours=12), eq + pd.Timedelta(hours=12), freq="1min", inclusive="left")
        sp = pvlib.solarposition.get_solarposition(times, lat, lon, method="nrel_numpy")
        vals.append(poa_daily(sp["zenith"].to_numpy(), sp["azimuth"].to_numpy(), tilt, surface_az, 1 / 60))
    return float(np.mean(vals))


@pytest.mark.parametrize("crs", ["EPSG:3338", "EPSG:3413"])
@pytest.mark.parametrize("lat,lon", SITES)
@pytest.mark.parametrize("slope,aspect", SURFACES)
def test_matches_pvlib_exact_equinox(crs, lat, lon, slope, aspect):
    dem, t = plane_patch(crs, lon, lat, slope, aspect)
    ours = centre(pisr(dem, t, crs, P))
    assert ours == pytest.approx(pvlib_exact(lat, slope, aspect), rel=2e-3)


@pytest.mark.parametrize("lat,lon", SITES)
@pytest.mark.parametrize("slope,aspect", SURFACES)
def test_close_to_pvlib_real_equinox_days(lat, lon, slope, aspect):
    dem, t = plane_patch("EPSG:3338", lon, lat, slope, aspect)
    ours = centre(pisr(dem, t, "EPSG:3338", P))
    assert ours == pytest.approx(pvlib_real(lat, lon, slope, aspect), rel=0.015)
