"""Vectorized SGP4 propagation and topocentric coordinate transforms.

Design notes
------------
* Uses SatrecArray (C extension) for batch SGP4 — one call per chunk of
  satellites × all timesteps, which is far faster than a Python loop.
* SGP4 returns positions in the TEME (True Equator Mean Equinox) frame,
  which is treated as quasi-ECI here.  The TEME–J2000 difference is a
  few arc-seconds — negligible for satellite visibility prediction.
* TEME → ECEF via GMST rotation (accurate to ~0.1″ for our purposes).
* ECEF → topocentric ENU → azimuth / elevation / slant range using WGS-84.

Assumptions
-----------
* Atmospheric refraction below ~5° elevation is *not* modelled.  Starlink
  connections below ~25° elevation are marginal regardless (high path loss,
  atmospheric effects), so sub-5° refraction corrections are irrelevant for
  service prediction.
* Observer altitude is assumed to be mean sea level (alt_km=0) unless
  explicitly provided.
"""

from __future__ import annotations

import numpy as np
from sgp4.api import Satrec, SatrecArray

# ---------------------------------------------------------------------------
# WGS-84 constants
# ---------------------------------------------------------------------------
WGS84_A = 6378.137          # semi-major axis, km
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2  # first eccentricity squared

# Earth gravitational parameter (km³ s⁻²)
MU_KM3_S2 = 398600.4418

# Earth mean radius for nadir-angle calculations (km)
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# TLE → Satrec list
# ---------------------------------------------------------------------------

def parse_tles(tles: list[dict]) -> tuple[list[Satrec], list[dict]]:
    """Parse TLE records into sgp4 Satrec objects.

    Silently drops any TLE that fails to parse (malformed checksum, etc.).

    Returns
    -------
    satrecs    : list of Satrec objects (same order as valid_tles)
    valid_tles : subset of input tles that parsed successfully
    """
    satrecs: list[Satrec] = []
    valid_tles: list[dict] = []
    for tle in tles:
        try:
            sat = Satrec.twoline2rv(tle["line1"], tle["line2"])
            satrecs.append(sat)
            valid_tles.append(tle)
        except Exception:
            continue
    return satrecs, valid_tles


# ---------------------------------------------------------------------------
# Observer position
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """Convert geodetic coordinates to WGS-84 ECEF (km).

    Parameters
    ----------
    lat_deg : geodetic latitude, degrees
    lon_deg : longitude, degrees east
    alt_km  : altitude above ellipsoid, km (default 0 = sea level)

    Returns
    -------
    np.ndarray shape (3,) — [x, y, z] in km
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + alt_km) * np.sin(lat)
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# Coordinate transforms (internal helpers)
# ---------------------------------------------------------------------------

def _gmst_rad(jd: np.ndarray) -> np.ndarray:
    """Greenwich Mean Sidereal Time in radians for an array of Julian dates.

    Accuracy: ~0.1 arc-second — sufficient for satellite visibility work.
    """
    T = (jd - 2451545.0) / 36525.0
    theta_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + T * T * (0.000387933 - T / 38710000.0)
    )
    return np.deg2rad(theta_deg % 360.0)


def _eci_to_ecef(r_eci: np.ndarray, jd: np.ndarray) -> np.ndarray:
    """Rotate ECI (TEME) position vectors to ECEF.

    Parameters
    ----------
    r_eci : shape (n_sats, n_times, 3), km
    jd    : shape (n_times,), combined Julian date (whole + fraction)

    Returns
    -------
    r_ecef : shape (n_sats, n_times, 3), km
    """
    theta = _gmst_rad(jd)        # (n_times,)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = r_eci[:, :, 0]          # (n_sats, n_times)
    y = r_eci[:, :, 1]
    z = r_eci[:, :, 2]

    r_ecef = np.empty_like(r_eci)
    r_ecef[:, :, 0] = x * cos_t + y * sin_t
    r_ecef[:, :, 1] = -x * sin_t + y * cos_t
    r_ecef[:, :, 2] = z
    return r_ecef


def _ecef_to_azel(
    r_ecef: np.ndarray,
    obs_ecef: np.ndarray,
    lat_deg: float,
    lon_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute topocentric azimuth, elevation, slant range from ECEF coords.

    Parameters
    ----------
    r_ecef   : shape (n_sats, n_times, 3), satellite ECEF positions, km
    obs_ecef : shape (3,), observer ECEF position, km
    lat_deg  : observer geodetic latitude, degrees
    lon_deg  : observer geodetic longitude, degrees

    Returns
    -------
    az_deg   : shape (n_sats, n_times), azimuth [0, 360) degrees
    el_deg   : shape (n_sats, n_times), elevation [-90, 90] degrees
    slant_km : shape (n_sats, n_times), slant range km
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # Range vector (satellite − observer) in ECEF
    dx = r_ecef[:, :, 0] - obs_ecef[0]   # (n_sats, n_times)
    dy = r_ecef[:, :, 1] - obs_ecef[1]
    dz = r_ecef[:, :, 2] - obs_ecef[2]

    slant_km = np.sqrt(dx * dx + dy * dy + dz * dz)

    # Rotate range vector to local East-North-Up frame
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    E =  -sin_lon * dx + cos_lon * dy
    N =  -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    U =   cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    el_rad = np.arcsin(np.clip(U / slant_km, -1.0, 1.0))
    az_rad = np.arctan2(E, N) % (2.0 * np.pi)

    return np.rad2deg(az_rad), np.rad2deg(el_rad), slant_km


# ---------------------------------------------------------------------------
# Main batch propagator
# ---------------------------------------------------------------------------

def propagate_batch(
    satrecs: list[Satrec],
    jd_whole: np.ndarray,
    jd_frac: np.ndarray,
    obs_ecef: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate all satellites over all timesteps; return topocentric coords.

    Satellites are processed in chunks of `chunk_size` to bound intermediate
    memory usage (the raw ECI position array for one chunk is ~17 MB for
    chunk_size=500, n_times=1441).

    SGP4 propagation failures (error code ≠ 0) are masked as NaN, which the
    pass-finder treats as below-horizon.

    Parameters
    ----------
    satrecs   : list of Satrec objects, length n_sats
    jd_whole  : shape (n_times,), integer part of Julian date
    jd_frac   : shape (n_times,), fractional part of Julian date
    obs_ecef  : shape (3,), observer ECEF position, km
    lat_deg   : observer geodetic latitude, degrees
    lon_deg   : observer geodetic longitude, degrees
    chunk_size: satellites per processing batch (tune for RAM vs speed)

    Returns
    -------
    az_deg   : shape (n_sats, n_times)
    el_deg   : shape (n_sats, n_times)
    slant_km : shape (n_sats, n_times)
    """
    n_sats = len(satrecs)
    n_times = len(jd_whole)
    jd_combined = jd_whole + jd_frac   # (n_times,) — used for GMST

    az_all = np.empty((n_sats, n_times), dtype=np.float64)
    el_all = np.empty((n_sats, n_times), dtype=np.float64)
    sl_all = np.empty((n_sats, n_times), dtype=np.float64)

    for start in range(0, n_sats, chunk_size):
        end = min(start + chunk_size, n_sats)
        sz = end - start

        # Build SatrecArray for this chunk.
        # SatrecArray.sgp4 takes 1-D jd/fr arrays and broadcasts across all
        # satellites internally — do NOT pass 2-D matrices here.
        chunk_array = SatrecArray(satrecs[start:end])

        # Vectorised SGP4: e shape (sz, n_times), r shape (sz, n_times, 3)
        e, r, _ = chunk_array.sgp4(jd_whole, jd_frac)

        # Mask propagation failures as NaN (non-zero error code)
        r[e != 0] = np.nan

        # TEME → ECEF → topocentric
        r_ecef = _eci_to_ecef(r, jd_combined)
        az, el, sl = _ecef_to_azel(r_ecef, obs_ecef, lat_deg, lon_deg)

        az_all[start:end] = az
        el_all[start:end] = el
        sl_all[start:end] = sl

    return az_all, el_all, sl_all


# ---------------------------------------------------------------------------
# Orbital shell helper
# ---------------------------------------------------------------------------

def orbital_altitude_km(line2: str) -> float:
    """Estimate mean orbital altitude (km) from TLE line 2 mean motion.

    Uses the vis-viva / Kepler third-law relation:
        a = (μ / n²)^(1/3)
    where n is mean motion in rad/s.  For Starlink's near-circular orbits
    the mean semi-major axis is a good proxy for actual altitude.
    """
    mean_motion_rev_day = float(line2[52:63])
    n_rad_s = mean_motion_rev_day * 2.0 * np.pi / 86400.0
    a_km = (MU_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)
    return a_km - WGS84_A
