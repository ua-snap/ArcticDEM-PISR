"""Clear-sky potential incoming solar radiation (PISR) for an idealized equinox day.

Model (per cell; local shading only, no cast shadows from distant terrain):
  * Terrain gradient (p = dz/dx, q = dz/dy with y grid-north positive) from Horn (1981) 3x3 differences.
  * Sun at declination 0 and hour angle w; true-frame unit vector (east, north, up):
        s = (-sin w, -sin(lat) cos w, cos(lat) cos w)
    The sun is above the horizon for w in (-90, 90) degrees at every latitude.
  * Projection geometry: the grid gradient is converted to a TRUE east/north gradient with the local
    Jacobian of the projection, J = d(x, y)/d(east, north) in metres per metre (from exact 1 m geodesic
    offsets):  g_east = p*dx/dE + q*dy/dE,  g_north = p*dx/dN + q*dy/dN.  This handles meridian convergence
    (tens of degrees in EPSG:3338 and ~85-140 degrees in EPSG:3413 over Alaska), projection scale (up to ~8% in
    3413, ~2.5% in 3338), and the angular distortion of non-conformal projections (up to ~3 degrees in Albers 3338).
    Surface unit normal (true frame) n = (-g_east, -g_north, 1) / sqrt(1 + g_east^2 + g_north^2).
  * Direct:  I0 * tau^m * max(cos i, 0), where cos i = n . s and m = Kasten & Young (1989) air mass.
  * Diffuse: I0 * sin(h) * max(0.271 - 0.294 tau^m, 0) * (1 + cos(slope)) / 2
    (Liu & Jordan 1960 diffuse transmittance; isotropic sky-view factor). Ground reflection excluded.
  * Daily total: midpoint-rule sum over the day with step `step_minutes`, in kWh m-2 day-1.
No pressure/elevation correction (constant lumped transmittance), so elevation enters only via slope/aspect.
"""
from dataclasses import asdict, dataclass

import numba as nb
import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class SolarParams:
    solar_constant: float = 1361.0  # W m-2 (Kopp & Lean 2011)
    transmittance: float = 0.70     # lumped clear-sky atmospheric transmittance (SAGA default)
    step_minutes: float = 5.0
    include_diffuse: bool = True

    def as_attrs(self):
        return {f"pisr_{k}": (int(v) if isinstance(v, bool) else v) for k, v in asdict(self).items()}


def hour_angles(step_minutes):
    """Midpoints of equal hour-angle steps over the daylight period (-90..90 deg) at declination 0."""
    step = np.deg2rad(step_minutes / 4.0)  # 1 minute of time = 0.25 deg of hour angle
    n = int(round(np.pi / step))
    return -np.pi / 2 + (np.arange(n) + 0.5) * (np.pi / n), (12.0 / n)  # (radians, hours per step)


@nb.njit(parallel=True, fastmath=False, cache=True)
def _pisr_kernel(dem, dx, dy, lat, jxe, jye, jxn, jyn, omegas, dt_h, i0, tau, include_diffuse):
    ny, nx = lat.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    ln_tau = np.log(tau)
    sin_w = np.sin(omegas)
    cos_w = np.cos(omegas)
    nsteps = omegas.size
    for r in nb.prange(ny):
        for c in range(nx):
            a = dem[r, c]; b = dem[r, c + 1]; cc = dem[r, c + 2]
            d = dem[r + 1, c]; f = dem[r + 1, c + 2]
            g = dem[r + 2, c]; h = dem[r + 2, c + 1]; ii = dem[r + 2, c + 2]
            e = dem[r + 1, c + 1]
            if (np.isnan(a) or np.isnan(b) or np.isnan(cc) or np.isnan(d) or np.isnan(e) or np.isnan(f)
                    or np.isnan(g) or np.isnan(h) or np.isnan(ii)):
                continue
            # Rows increase southward in the grid, so north-positive gradient is (top - bottom).
            p = ((cc + 2.0 * f + ii) - (a + 2.0 * d + g)) / (8.0 * dx)
            q = ((a + 2.0 * b + cc) - (g + 2.0 * h + ii)) / (8.0 * dy)
            ge = p * jxe[r, c] + q * jye[r, c]  # true east/north gradient (chain rule through the projection)
            gn = p * jxn[r, c] + q * jyn[r, c]
            inv_norm = 1.0 / np.sqrt(1.0 + ge * ge + gn * gn)  # = cos(true slope)
            svf = 0.5 * (1.0 + inv_norm)
            sphi = np.sin(lat[r, c]); cphi = np.cos(lat[r, c])
            total = 0.0
            for k in range(nsteps):
                sz = cphi * cos_w[k]
                if sz <= 0.0:
                    continue
                se = -sin_w[k]
                sn = -sphi * cos_w[k]
                cosi = (-ge * se - gn * sn + sz) * inv_norm
                h_deg = np.degrees(np.arcsin(sz))
                m = 1.0 / (sz + 0.50572 * (h_deg + 6.07995) ** -1.6364)
                tb = np.exp(m * ln_tau)
                g_tot = 0.0
                if cosi > 0.0:
                    g_tot += i0 * tb * cosi
                if include_diffuse:
                    td = 0.271 - 0.294 * tb
                    if td > 0.0:
                        g_tot += i0 * sz * td * svf
                total += g_tot
            out[r, c] = total * dt_h / 1000.0
    return out


def lat_and_jacobian(transform, shape, crs, stride=16, step_m=10.0):
    """Latitude (rad) and the projection Jacobian (dx/dE, dy/dE, dx/dN, dy/dN) at every cell centre.

    Computed exactly on a coarse lattice (every `stride` cells) with geodesic +/- `step_m` offsets due east and
    due north on the CRS's ellipsoid (central differences), then bilinearly interpolated (smooth fields).
    The grid azimuth of true north is atan2(dx/dN, dy/dN).
    """
    ny, nx = shape
    rows = np.unique(np.r_[np.arange(0, ny, stride), ny - 1])
    cols = np.unique(np.r_[np.arange(0, nx, stride), nx - 1])
    cc, rr = np.meshgrid(cols + 0.5, rows + 0.5)
    x = transform.c + cc * transform.a + rr * transform.b
    y = transform.f + cc * transform.d + rr * transform.e
    crs = CRS.from_user_input(crs)
    geod = crs.get_geod()
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    fwd = Transformer.from_crs(4326, crs, always_xy=True)
    lon, lat = inv.transform(x, y)
    jac = []
    for az in (90.0, 0.0):  # east, north
        lon_p, lat_p, _ = geod.fwd(lon, lat, np.full_like(lon, az), np.full_like(lon, step_m))
        lon_m, lat_m, _ = geod.fwd(lon, lat, np.full_like(lon, az + 180.0), np.full_like(lon, step_m))
        xp, yp = fwd.transform(lon_p, lat_p)
        xm, ym = fwd.transform(lon_m, lat_m)
        jac += [(xp - xm) / (2 * step_m), (yp - ym) / (2 * step_m)]
    fields = [np.deg2rad(lat)] + jac  # lat, dx/dE, dy/dE, dx/dN, dy/dN
    return tuple(_bilinear(v, rows, cols, ny, nx) for v in fields)


def _bilinear(v, rows, cols, ny, nx):
    ri = np.interp(np.arange(ny), rows, np.arange(rows.size))
    ci = np.interp(np.arange(nx), cols, np.arange(cols.size))
    r0 = np.minimum(np.floor(ri).astype(int), rows.size - 2) if rows.size > 1 else np.zeros(ny, int)
    c0 = np.minimum(np.floor(ci).astype(int), cols.size - 2) if cols.size > 1 else np.zeros(nx, int)
    if rows.size == 1 or cols.size == 1:
        return np.broadcast_to(v.mean(), (ny, nx)).astype(np.float64).copy()
    fr = (ri - r0)[:, None]
    fc = (ci - c0)[None, :]
    v00 = v[r0][:, c0]; v01 = v[r0][:, c0 + 1]; v10 = v[r0 + 1][:, c0]; v11 = v[r0 + 1][:, c0 + 1]
    return (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc + v10 * fr * (1 - fc) + v11 * fr * fc)


def pisr(dem_halo, transform, crs, params=SolarParams(), geo=None):
    """PISR (kWh m-2 day-1) for the interior of `dem_halo` (1-cell halo on every side; NaN = nodata).

    `transform` is the affine transform of the interior (output) grid. `geo` optionally supplies precomputed
    (lat, dx/dE, dy/dE, dx/dN, dy/dN) arrays for the interior, as returned by `lat_and_jacobian`.
    """
    ny, nx = dem_halo.shape[0] - 2, dem_halo.shape[1] - 2
    if geo is None:
        geo = lat_and_jacobian(transform, (ny, nx), crs)
    lat, jxe, jye, jxn, jyn = (np.ascontiguousarray(g, dtype=np.float64) for g in geo)
    omegas, dt_h = hour_angles(params.step_minutes)
    return _pisr_kernel(np.ascontiguousarray(dem_halo, dtype=np.float64), abs(transform.a), abs(transform.e),
                        lat, jxe, jye, jxn, jyn, omegas, dt_h,
                        params.solar_constant, params.transmittance, params.include_diffuse)


def reference_flat(lat_deg, params=SolarParams()):
    """PISR (kWh m-2 day-1) of a horizontal surface at each latitude in `lat_deg` (1-D), via the same kernel.

    At latitude 0 this is an upper bound for any surface anywhere (see qc.py).
    """
    lat = np.deg2rad(np.atleast_1d(np.asarray(lat_deg, dtype=np.float64)))[:, None]
    n = lat.shape[0]
    one, zero = np.ones((n, 1)), np.zeros((n, 1))
    omegas, dt_h = hour_angles(params.step_minutes)
    out = _pisr_kernel(np.zeros((n + 2, 3)), 1.0, 1.0, lat, one, zero, zero, one, omegas, dt_h,
                       params.solar_constant, params.transmittance, params.include_diffuse)
    return out[:, 0].astype(np.float64)


def terrain_insolation_index(dem_halo, transform, crs, params=SolarParams(), ref_lat_deg=64.0, geo=None):
    """Terrain insolation index (TII), kWh m-2 day-1: slope/aspect effect with latitude removed.

    TII = PISR computed as if the cell were at `ref_lat_deg` (true slope and true aspect kept, via the projection
    Jacobian) minus PISR of flat ground at `ref_lat_deg`. Flat ground = 0, warm (sun-facing) slopes > 0, cold slopes
    < 0. Identical slope/aspect gives the identical value anywhere in the domain.
    """
    ny, nx = dem_halo.shape[0] - 2, dem_halo.shape[1] - 2
    if geo is None:
        geo = lat_and_jacobian(transform, (ny, nx), crs)
    geo = (np.full((ny, nx), np.deg2rad(ref_lat_deg)),) + tuple(geo[1:])
    flat = reference_flat([ref_lat_deg], params)[0]
    return (pisr(dem_halo, transform, crs, params, geo=geo) - flat).astype(np.float32)
