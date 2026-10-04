"""Pass search: horizon mask interpolation, 1 s grid scan, 0.1 s bisection."""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sgp4.api import jday

from .coords import ecef_to_enu_az_el_range, geodetic_to_ecef, teme_to_ecef

GRID_STEP_S = 1.0
BISECT_TOL_S = 0.1


class PropagationError(RuntimeError):
    pass


class HorizonMask:
    """Min-elevation mask as sorted (azimuth_deg, elevation_deg) nodes.

    Linear interpolation in azimuth, wrapping across 0/360 deg.
    """

    def __init__(self, nodes: list[tuple[float, float]] | None):
        if not nodes:
            self._az = [0.0]
            self._el = [0.0]
            return
        pts = sorted((float(a) % 360.0, float(e)) for a, e in nodes)
        self._az = [p[0] for p in pts]
        self._el = [p[1] for p in pts]

    def min_elevation(self, az_deg: float) -> float:
        az = az_deg % 360.0
        azs, els = self._az, self._el
        n = len(azs)
        if n == 1:
            return els[0]
        i = bisect.bisect_right(azs, az)
        a0, e0 = azs[i - 1], els[i - 1]
        if i == 0:
            a0 -= 360.0
        a1, e1 = (azs[i], els[i]) if i < n else (azs[0] + 360.0, els[0])
        if a1 == a0:
            return e1
        frac = (az - a0) / (a1 - a0)
        return e0 + frac * (e1 - e0)


@dataclass
class Site:
    station_id: str
    lat_deg: float
    lon_deg: float
    alt_m: float
    mask: HorizonMask
    ecef: tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        self.ecef = geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m)


@dataclass
class LookAngles:
    az_deg: float
    el_deg: float
    range_km: float


def _propagate(satrec, dt: datetime):
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                  dt.second + dt.microsecond / 1e6)
    err, r, v = satrec.sgp4(jd, fr)
    if err != 0:
        raise PropagationError(
            f"SGP4 propagation failed (error {err}) at {dt.isoformat()}")
    return r, v


def look_angles(satrec, site: Site, dt: datetime) -> LookAngles:
    r_teme, _ = _propagate(satrec, dt)
    sat_ecef = teme_to_ecef(r_teme, dt)
    az, el, rng = ecef_to_enu_az_el_range(
        sat_ecef, site.ecef, site.lat_deg, site.lon_deg)
    return LookAngles(az, el, rng)


def _margin(satrec, site: Site, dt: datetime) -> float:
    la = look_angles(satrec, site, dt)
    return la.el_deg - site.mask.min_elevation(la.az_deg)


def _bisect_crossing(satrec, site, t_before, t_after) -> datetime:
    """t_before/t_after straddle a sign change of the margin (either
    direction); refine the crossing and return the visible-side estimate."""
    lo, hi = t_before, t_after
    lo_visible = _margin(satrec, site, lo) > 0.0
    while (hi - lo).total_seconds() > BISECT_TOL_S:
        mid = lo + (hi - lo) / 2
        if (_margin(satrec, site, mid) > 0.0) == lo_visible:
            lo = mid
        else:
            hi = mid
    return lo if lo_visible else hi


@dataclass
class PassInterval:
    start: datetime
    end: datetime
    duration_s: float
    truncated_at_start: bool
    truncated_at_end: bool
    max_el_deg: float
    max_el_time: datetime


def find_passes(satrec, site: Site, start: datetime, end: datetime
                ) -> list[PassInterval]:
    """Intervals where elevation strictly exceeds the horizon mask.

    Tangential touches (margin == 0 without sign change) never form an
    interval. Intervals shorter than the 1 s grid may be missed.
    """
    # 1 s grid that also covers the fractional tail of the window
    span = (end - start).total_seconds()
    n = int(span)
    times = [start + timedelta(seconds=i) for i in range(n + 1)]
    if times[-1] < end:
        times.append(end)
        n += 1
    margins = [_margin(satrec, site, t) for t in times]
    visible = [m > 0.0 for m in margins]

    intervals: list[PassInterval] = []
    i = 0
    while i <= n:
        if not visible[i]:
            i += 1
            continue
        j = i
        while j + 1 <= n and visible[j + 1]:
            j += 1
        # rising edge between i-1 and i (unless at window start)
        if i == 0:
            rise, trunc_start = times[0], True
        else:
            rise = _bisect_crossing(satrec, site, times[i - 1], times[i])
            trunc_start = False
        if j == n:
            fall, trunc_end = times[n], True
        else:
            fall = _bisect_crossing(satrec, site, times[j], times[j + 1])
            trunc_end = False
        duration = (fall - rise).total_seconds()
        if duration > 0.0:
            # max elevation sampled on the 1 s grid inside the interval
            best_t, best_el = None, -math.inf
            for t in times[i:j + 1]:
                la = look_angles(satrec, site, t)
                if la.el_deg > best_el:
                    best_el, best_t = la.el_deg, t
            intervals.append(PassInterval(
                start=rise, end=fall, duration_s=duration,
                truncated_at_start=trunc_start, truncated_at_end=trunc_end,
                max_el_deg=best_el, max_el_time=best_t))
        i = j + 1
    return intervals
