"""Coordinate transforms.

Approximations (documented in README):
- UTC is used in place of UT1 (|UTC-UT1| < 0.9 s).
- TEME is rotated to ECEF with GMST only; nutation, polar motion and
  the TEME->PEF pseudo-fixed offsets are neglected.
- WGS84 geodetic site coordinates; SGP4 propagates in WGS72.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

# WGS84 ellipsoid
WGS84_A = 6378137.0            # semi-major axis, m
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

EARTH_OMEGA = 7.2921150e-5     # rad/s, inertial rotation rate
SPEED_OF_LIGHT = 299792458.0   # m/s


def julian_date(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def gmst_rad(dt: datetime) -> float:
    """Greenwich mean sidereal time (IAU 1982), UTC approximating UT1."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    deg = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
           + 0.000387933 * t * t - t ** 3 / 38710000.0)
    return math.radians(deg % 360.0)


def teme_to_ecef(r_teme: tuple[float, float, float], dt: datetime
                 ) -> tuple[float, float, float]:
    """Rotate TEME (km) into the rotating frame by -GMST about z."""
    th = gmst_rad(dt)
    c, s = math.cos(th), math.sin(th)
    x, y, z = r_teme
    return (c * x + s * y, -s * x + c * y, z)


def teme_vel_to_ecef(r_teme, v_teme, dt):
    """TEME velocity (km/s) -> ECEF velocity, subtracting omega x r."""
    th = gmst_rad(dt)
    c, s = math.cos(th), math.sin(th)
    vx, vy, vz = v_teme
    ex, ey, ez = c * vx + s * vy, -s * vx + c * vy, vz
    rx, ry, _ = teme_to_ecef(r_teme, dt)
    return (ex + EARTH_OMEGA * ry, ey - EARTH_OMEGA * rx, ez)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float
                     ) -> tuple[float, float, float]:
    """WGS84 geodetic -> ECEF, returned in km."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return (x / 1000.0, y / 1000.0, z / 1000.0)


def ecef_to_enu_az_el_range(sat_ecef, site_ecef, lat_deg, lon_deg):
    """Azimuth (deg, clockwise from north), elevation (deg), range (km)."""
    dx = sat_ecef[0] - site_ecef[0]
    dy = sat_ecef[1] - site_ecef[1]
    dz = sat_ecef[2] - site_ecef[2]
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    rng = math.sqrt(east * east + north * north + up * up)
    el = math.degrees(math.asin(up / rng)) if rng > 0 else 90.0
    az = math.degrees(math.atan2(east, north)) % 360.0
    return az, el, rng
