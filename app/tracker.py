"""Dual-axis mechanical tracking planner.

Generates 1 s look-angle targets (interval endpoints included) with the
existing propagation/site geometry, then picks mechanical azimuth
unwrapping (azimuth + 360*k, no over-the-top flip) so that the complete
path current -> preset -> tracking -> homing respects position and rate
limits. Among feasible paths the one with minimal total azimuth travel
wins; ties are broken by lexicographic mechanical azimuth sequence.
Infeasible intervals are rejected as a whole (no clipping, no skipping).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from .passes import Site, look_angles

GRID_STEP_S = 1.0
MAX_INTERVAL_S = 1800.0  # a single tracked segment must not exceed 30 min


class PlanError(ValueError):
    """Raised when no feasible mechanical path exists."""


@dataclass
class AxisLimits:
    az_min_deg: float
    az_max_deg: float
    el_min_deg: float
    el_max_deg: float
    max_az_rate_dps: float
    max_el_rate_dps: float


@dataclass
class MechTarget:
    t_rel_s: float      # seconds relative to the first tracking target
    az_deg: float       # mechanical azimuth (may exceed [0, 360))
    el_deg: float


@dataclass
class TrackPlan:
    targets: list[MechTarget]
    total_az_travel_deg: float


def target_times(start: datetime, end: datetime) -> list[datetime]:
    """1 s grid from start to end, both endpoints included."""
    times = []
    t = start
    while t < end:
        times.append(t)
        t += timedelta(seconds=GRID_STEP_S)
    if not times or times[-1] < end:
        times.append(end)
    return times


def _az_candidates(az_deg: float, lo: float, hi: float) -> list[float]:
    """All az_deg + 360*k values inside [lo, hi], ascending."""
    k0 = math.ceil((lo - az_deg) / 360.0 - 1e-9)
    k1 = math.floor((hi - az_deg) / 360.0 + 1e-9)
    return [az_deg + 360.0 * k for k in range(k0, k1 + 1)]


def plan_track(satrec, site: Site, start: datetime, end: datetime,
               limits: AxisLimits,
               current: tuple[float, float],
               home: tuple[float, float],
               preset_s: float, homing_s: float) -> TrackPlan:
    """Plan a full mechanical path for one visibility interval."""
    duration = (end - start).total_seconds()
    if duration <= 0.0:
        raise PlanError("interval has no positive duration")
    if duration > MAX_INTERVAL_S + 1e-9:
        raise PlanError(
            f"interval of {duration:.1f} s exceeds the 30 min segment limit")

    times = target_times(start, end)
    looks = [look_angles(satrec, site, t) for t in times]
    n = len(times)

    # elevation: fixed per point, must stay inside limits the whole time
    els = [la.el_deg for la in looks]
    for t, el in zip(times, els):
        if not (limits.el_min_deg - 1e-9 <= el <= limits.el_max_deg + 1e-9):
            raise PlanError(
                f"elevation {el:.3f} deg at {t.isoformat()} outside "
                f"[{limits.el_min_deg}, {limits.el_max_deg}]")

    def el_rate_ok(el_a: float, el_b: float, dt_s: float) -> bool:
        return abs(el_b - el_a) <= limits.max_el_rate_dps * dt_s + 1e-9

    # preset / homing elevation feasibility (independent of azimuth choice)
    if not el_rate_ok(current[1], els[0], preset_s):
        raise PlanError("preset elevation slew exceeds the rate limit")
    if not el_rate_ok(els[-1], home[1], homing_s):
        raise PlanError("homing elevation slew exceeds the rate limit")
    for i in range(n - 1):
        dt_s = (times[i + 1] - times[i]).total_seconds()
        if not el_rate_ok(els[i], els[i + 1], dt_s):
            raise PlanError(
                f"elevation rate limit exceeded near {times[i].isoformat()}")

    # per-point mechanical azimuth candidates (azimuth + 360*k in limits)
    cand = []
    for la, t in zip(looks, times):
        opts = _az_candidates(la.az_deg, limits.az_min_deg, limits.az_max_deg)
        if not opts:
            raise PlanError(
                f"azimuth {la.az_deg:.3f} deg at {t.isoformat()} cannot be "
                f"reached within [{limits.az_min_deg}, {limits.az_max_deg}]")
        cand.append(opts)

    # dynamic programming over candidate choices; state value is
    # (min total azimuth travel so far, lexicographically smallest sequence)
    prev: dict[float, tuple[float, tuple[float, ...]]] = {}
    for a in cand[0]:
        if abs(a - current[0]) <= limits.max_az_rate_dps * preset_s + 1e-9:
            prev[a] = (abs(a - current[0]), (a,))
    if not prev:
        raise PlanError("preset azimuth slew exceeds the rate limit")

    for i in range(1, n):
        dt_s = (times[i] - times[i - 1]).total_seconds()
        max_step = limits.max_az_rate_dps * dt_s + 1e-9
        nxt: dict[float, tuple[float, tuple[float, ...]]] = {}
        for a in cand[i]:
            best = None
            for a_prev, (cost, seq) in prev.items():
                step = abs(a - a_prev)
                if step > max_step:
                    continue
                val = (cost + step, seq + (a,))
                if best is None or val < best:
                    best = val
            if best is not None:
                nxt[a] = best
        if not nxt:
            raise PlanError(
                f"azimuth rate/limit infeasible near {times[i].isoformat()}")
        prev = nxt

    best = None
    for a, (cost, seq) in prev.items():
        step = abs(home[0] - a)
        if step > limits.max_az_rate_dps * homing_s + 1e-9:
            continue
        val = (cost + step, seq)
        if best is None or val < best:
            best = val
    if best is None:
        raise PlanError("homing azimuth slew exceeds the rate limit")

    total_cost, seq = best
    t0 = times[0]
    targets = [MechTarget(t_rel_s=(t - t0).total_seconds(),
                          az_deg=a, el_deg=el)
               for t, a, el in zip(times, seq, els)]
    return TrackPlan(targets=targets, total_az_travel_deg=total_cost)

