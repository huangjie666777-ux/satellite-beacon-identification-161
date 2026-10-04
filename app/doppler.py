"""Doppler shift modelling, phase correction and per-window diagnostics.

Radial velocity reuses the same SGP4 propagation and site geometry as the
forecast CSV (ECEF range rate, positive when the satellite recedes).
The baseband frequency shift is

    shift(t) = (f_tx - f_center) - f_tx * range_rate(t) / c

i.e. where the transmitted carrier appears in the recorded baseband.
Correction multiplies samples by exp(-j * phi(t)) with phi(0) = 0 and
phi the time integral of shift; no resampling, no normalisation.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .coords import SPEED_OF_LIGHT, teme_to_ecef, teme_vel_to_ecef
from .passes import Site, _propagate

NODE_STEP_S = 1.0
CORRECTION_BLOCK = 1 << 16
WINDOW_LEN = 1024


def range_rate_mps(satrec, site: Site, dt: datetime) -> float:
    """ECEF range rate (m/s), positive when the range increases."""
    r_teme, v_teme = _propagate(satrec, dt)
    sat_ecef = teme_to_ecef(r_teme, dt)
    vel_ecef = teme_vel_to_ecef(r_teme, v_teme, dt)
    dx = sat_ecef[0] - site.ecef[0]
    dy = sat_ecef[1] - site.ecef[1]
    dz = sat_ecef[2] - site.ecef[2]
    rng = (dx * dx + dy * dy + dz * dz) ** 0.5
    rdot_kmps = (dx * vel_ecef[0] + dy * vel_ecef[1]
                 + dz * vel_ecef[2]) / rng
    return rdot_kmps * 1000.0


def shift_node_times(duration_s: float) -> np.ndarray:
    """1 s node grid starting at 0, always including the exact end."""
    n_full = int(duration_s / NODE_STEP_S)
    times = [k * NODE_STEP_S for k in range(n_full + 1)]
    if not times or times[-1] < duration_s:
        times.append(duration_s)
    return np.array(times, dtype=np.float64)


def shift_nodes(satrec, site: Site, start: datetime, node_times_s,
                tx_hz: float, center_hz: float) -> np.ndarray:
    """Baseband shift (Hz) of the transmitted carrier at each node."""
    shifts = []
    for t in node_times_s:
        dt = start + timedelta(seconds=float(t))
        rdot = range_rate_mps(satrec, site, dt)
        shifts.append((tx_hz - center_hz) - tx_hz * rdot / SPEED_OF_LIGHT)
    return np.array(shifts, dtype=np.float64)


def correct(samples: np.ndarray, sample_rate_hz: float,
            node_times_s: np.ndarray, node_shifts_hz: np.ndarray
            ) -> np.ndarray:
    """Multiply by exp(-j*phi(t)), phi(0)=0, phi' = shift; block-wise
    with continuous phase across blocks. Input/output are complex64."""
    out = np.empty_like(samples)
    phase = 0.0
    prev_shift: float | None = None
    pos = 0
    total = len(samples)
    inv_fs = 1.0 / sample_rate_hz
    while pos < total:
        end = min(pos + CORRECTION_BLOCK, total)
        t = np.arange(pos, end, dtype=np.float64) * inv_fs
        shift = np.interp(t, node_times_s, node_shifts_hz)
        increments = np.empty(end - pos, dtype=np.float64)
        if prev_shift is None:
            increments[0] = 0.0  # phase is exactly 0 at the first sample
        else:
            increments[0] = 0.5 * (prev_shift + shift[0]) * inv_fs
        increments[1:] = 0.5 * (shift[:-1] + shift[1:]) * inv_fs
        phases = phase + np.cumsum(increments)
        rot = np.exp(-2j * np.pi * phases).astype(np.complex64)
        out[pos:end] = samples[pos:end] * rot
        phase = float(phases[-1])
        prev_shift = float(shift[-1])
        pos = end
    return out


def window_diagnostics(before: np.ndarray, after: np.ndarray,
                       sample_rate_hz: float,
                       window_len: int = WINDOW_LEN) -> list[dict]:
    """Per non-overlapping Hann-window peak frequency (Hz, signed) and
    mean-square power, before and after correction. Empty when the
    recording is shorter than one window."""
    count = len(before) // window_len
    if count == 0:
        return []
    window = np.hanning(window_len)
    freqs = np.fft.fftfreq(window_len, d=1.0 / sample_rate_hz)
    rows = []
    for k in range(count):
        seg_b = before[k * window_len:(k + 1) * window_len]
        seg_a = after[k * window_len:(k + 1) * window_len]
        peak_b = float(freqs[int(np.argmax(np.abs(np.fft.fft(seg_b * window))))])
        peak_a = float(freqs[int(np.argmax(np.abs(np.fft.fft(seg_a * window))))])
        rows.append({
            "index": k,
            "t_start_s": round(k * window_len / sample_rate_hz, 9),
            "peak_freq_hz_before": peak_b,
            "peak_freq_hz_after": peak_a,
            "mean_power_before": float(np.mean(np.abs(seg_b) ** 2)),
            "mean_power_after": float(np.mean(np.abs(seg_a) ** 2)),
        })
    return rows
