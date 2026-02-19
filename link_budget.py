"""Starlink link likelihood estimation from pass geometry.

Background
----------
Starlink satellites use electronically steered phased-array antennas.
Each satellite can serve ground users within a cone of roughly 25° half-angle
measured from nadir (the point directly below the satellite).  Passes where
the observer lies inside this cone are flagged "likely"; those on the edge
are "marginal"; those outside are "unlikely".

The nadir angle θ from the satellite to the observer is computed via the
spherical-Earth triangle relation:

    sin(θ) = R_e · cos(ε) / (R_e + h)

where ε is the observer elevation angle to the satellite and h is the orbital
altitude.

Elevation also matters independently: even if a satellite is geometrically
overhead, Starlink connections below ~25° elevation suffer significant
atmospheric attenuation and multipath.  Such passes are downgraded to
"marginal" regardless of beam geometry.

Shells modelled
---------------
V1  (Gen-1)    ~540–570 km, inclination ~53°
V1.5 Polar     ~540–580 km, high-inclination shells
V2 Mini        ~515–545 km, multiple inclination planes

References: Starlink FCC filings; Bhattacherjee et al. 2019; public
satellite tracking databases.
"""

from __future__ import annotations

import numpy as np

# Half-angle of Starlink user beam from nadir (degrees).
# Based on regulatory filings and link-geometry analyses; actual value
# varies slightly by satellite generation and operating mode.
BEAM_HALF_ANGLE_DEG: float = 25.0

# Elevation below which a connection is considered marginal regardless of
# beam footprint.  Low-elevation passes face multipath, blockage, and
# atmospheric loss well before the beam edge becomes relevant.
MIN_USEFUL_EL_DEG: float = 25.0

# Spherical Earth radius used for nadir-angle geometry (km).
EARTH_RADIUS_KM: float = 6371.0


# ---------------------------------------------------------------------------
# Shell classification
# ---------------------------------------------------------------------------

def classify_shell(altitude_km: float) -> str:
    """Return a human-readable Starlink shell label for an orbital altitude."""
    if 515 <= altitude_km <= 545:
        return f"V2 Mini (~{altitude_km:.0f} km)"
    if 540 <= altitude_km <= 580:
        return f"V1/V1.5 (~{altitude_km:.0f} km)"
    if altitude_km < 515:
        return f"Low orbit ({altitude_km:.0f} km)"
    if altitude_km < 620:
        return f"Mid orbit ({altitude_km:.0f} km)"
    return f"High orbit ({altitude_km:.0f} km)"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def nadir_angle_deg(elevation_deg: float, altitude_km: float) -> float:
    """Satellite nadir angle toward the observer (degrees).

    This is the angle at the satellite between the sub-satellite point and
    the line of sight to the observer.  If nadir_angle < BEAM_HALF_ANGLE_DEG
    the observer is within the satellite's main beam footprint.
    """
    el = np.deg2rad(elevation_deg)
    sin_nadir = EARTH_RADIUS_KM * np.cos(el) / (EARTH_RADIUS_KM + altitude_km)
    sin_nadir = float(np.clip(sin_nadir, -1.0, 1.0))
    return float(np.rad2deg(np.arcsin(sin_nadir)))


# ---------------------------------------------------------------------------
# Link likelihood
# ---------------------------------------------------------------------------

def estimate_link_likelihood(
    peak_el_deg: float,
    orbital_alt_km: float,
) -> tuple[str, str]:
    """Estimate the probability of a Starlink connection during a pass.

    Uses peak elevation and nadir angle at peak as proxy for whether the
    observer is likely within the satellite's serving beam.

    Returns
    -------
    (likelihood, note)
        likelihood : "likely" | "marginal" | "unlikely"
        note       : short human-readable explanation
    """
    nadir = nadir_angle_deg(peak_el_deg, orbital_alt_km)
    in_beam = nadir < BEAM_HALF_ANGLE_DEG
    near_edge = nadir < BEAM_HALF_ANGLE_DEG + 10.0
    low_el = peak_el_deg < MIN_USEFUL_EL_DEG

    if in_beam and not low_el:
        return (
            "likely",
            f"In beam footprint (nadir {nadir:.1f}° < {BEAM_HALF_ANGLE_DEG}°), "
            f"el {peak_el_deg:.1f}°",
        )
    if in_beam and low_el:
        return (
            "marginal",
            f"In beam (nadir {nadir:.1f}°) but low elevation ({peak_el_deg:.1f}° "
            f"< {MIN_USEFUL_EL_DEG}°)",
        )
    if near_edge and not low_el:
        return (
            "marginal",
            f"Near beam edge (nadir {nadir:.1f}°, limit {BEAM_HALF_ANGLE_DEG}°), "
            f"el {peak_el_deg:.1f}°",
        )
    return (
        "unlikely",
        f"Outside beam footprint (nadir {nadir:.1f}° > {BEAM_HALF_ANGLE_DEG}°), "
        f"el {peak_el_deg:.1f}°",
    )
