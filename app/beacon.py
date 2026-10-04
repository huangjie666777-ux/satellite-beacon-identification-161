"""Unknown narrowband beacon identification.

The recording is expected to contain a single stable unmodulated carrier.
Identity annotations in the SigMF metadata are never consulted: candidates
are ranked purely by how well the measured carrier frequency track matches
the predicted baseband shift of each candidate downlink.

Measurement: non-overlapping 1024-point Hann windows; the spectral peak is
refined with a 3-point parabolic interpolation on log magnitude across
the FFT wrap boundary.  A window is valid only when its peak power
exceeds 10x the median spectral power of that window.  Each valid window
keeps its UTC centre time; the span of valid windows must cover at
least 5 s.

Per candidate the predicted shift at each window centre is
    shift(t) = (f_tx - f_center) - f_tx * range_rate(t) / c
reusing the same SGP4/ECEF geometry as the forecast.  The constant local
oscillator offset is estimated as mean(observed - predicted) and the
debiased residual RMS ranks the candidates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .coords import SPEED_OF_LIGHT
from .doppler import range_rate_mps
from .passes import Site, find_passes

WINDOW_LEN = 1024
PEAK_TO_MEDIAN_MIN = 10.0
MIN_SPAN_S = 5.0


class BeaconError(ValueError):
    """Raised when the recording cannot support identification."""


@dataclass
class WindowMeasurement:
    index: int
    time_utc: datetime
    t_offset_s: float          # window centre relative to recording start
    freq_hz: float             # refined peak frequency, signed baseband
    peak_power: float
    median_power: float
    accepted: bool


def measure_windows(samples: np.ndarray, sample_rate_hz: float,
                    start: datetime, window_len: int = WINDOW_LEN
                    ) -> list[WindowMeasurement]:
    """Per non-overlapping Hann-window refined peak frequency (Hz, signed).

    The parabolic refinement uses bins k-1, k, k+1 modulo the FFT length,
    so peaks at the wrap boundary (bins 0 and N-1) are handled correctly.
    """
    count = len(samples) // window_len
    if count == 0:
        return []
    window = np.hanning(window_len)
    bin_hz = sample_rate_hz / window_len
    out: list[WindowMeasurement] = []
    for k in range(count):
        seg = samples[k * window_len:(k + 1) * window_len] * window
        mag = np.abs(np.fft.fft(seg))
        power = mag * mag
        pk = int(np.argmax(power))
        median = float(np.median(power))
        left = math.log10(max(mag[(pk - 1) % window_len], 1e-30))
        centre = math.log10(max(mag[pk], 1e-30))
        right = math.log10(max(mag[(pk + 1) % window_len], 1e-30))
        denom = left - 2.0 * centre + right
        delta = 0.0
        if denom != 0.0:
            delta = 0.5 * (left - right) / denom
            delta = float(min(0.5, max(-0.5, delta)))
        frac_bin = (pk + delta) % window_len
        freq = frac_bin * bin_hz
        if freq >= sample_rate_hz / 2.0:
            freq -= sample_rate_hz
        t_off = (k * window_len + window_len / 2.0) / sample_rate_hz
        out.append(WindowMeasurement(
            index=k,
            time_utc=start + timedelta(seconds=t_off),
            t_offset_s=t_off,
            freq_hz=freq,
            peak_power=float(power[pk]),
            median_power=median,
            accepted=bool(power[pk] > PEAK_TO_MEDIAN_MIN * median)))
    return out


def valid_windows(measurements: list[WindowMeasurement]
                  ) -> list[WindowMeasurement]:
    """Accepted windows; raise when their time span is below 5 s."""
    valid = [m for m in measurements if m.accepted]
    if not valid:
        raise BeaconError(
            "no usable window: every 1024-point window failed the "
            "peak-to-median carrier test (no single stable carrier found)")
    span = valid[-1].t_offset_s - valid[0].t_offset_s
    if span < MIN_SPAN_S:
        raise BeaconError(
            f"valid windows span only {span:.2f} s, below the required "
            f"{MIN_SPAN_S:.0f} s")
    return valid


@dataclass
class CandidateResult:
    satellite_id: str
    status: str                       # "comparable" | "excluded"
    reason: str | None = None
    lo_offset_hz: float | None = None
    rms_residual_hz: float | None = None
    rows: list[dict] = field(default_factory=list)


def evaluate_candidate(satellite_id: str, satrec, site: Site,
                       rec_start: datetime, rec_end: datetime,
                       windows: list[WindowMeasurement],
                       tx_hz: float, center_hz: float,
                       sample_rate_hz: float,
                       lo_offset_limit_hz: float) -> CandidateResult:
    """Compare one candidate downlink against the measured track."""
    try:
        intervals = find_passes(satrec, site, rec_start, rec_end)
    except Exception as exc:  # propagation failure excludes the candidate
        return CandidateResult(satellite_id, "excluded",
                               reason=f"propagation failed: {exc}")
    if not any(iv.start <= rec_start and iv.end >= rec_end
               for iv in intervals):
        return CandidateResult(
            satellite_id, "excluded",
            reason="recording is not fully inside a single visibility "
                   "interval for this candidate")
    nyquist = sample_rate_hz / 2.0
    observed = np.array([w.freq_hz for w in windows], dtype=np.float64)
    predicted = np.empty(len(windows), dtype=np.float64)
    for i, w in enumerate(windows):
        rdot = range_rate_mps(satrec, site, w.time_utc)
        predicted[i] = (tx_hz - center_hz) - tx_hz * rdot / SPEED_OF_LIGHT
    worst = float(np.max(np.abs(predicted)))
    if worst >= nyquist:
        return CandidateResult(
            satellite_id, "excluded",
            reason=f"predicted baseband shift reaches {worst:.1f} Hz, not "
                   f"below half the sample rate ({nyquist:.1f} Hz)")
    offsets = observed - predicted
    lo_offset = float(np.mean(offsets))
    if abs(lo_offset) > lo_offset_limit_hz:
        return CandidateResult(
            satellite_id, "excluded",
            reason=f"estimated LO offset {lo_offset:.1f} Hz exceeds the "
                   f"limit {lo_offset_limit_hz:.1f} Hz")
    residuals = offsets - lo_offset
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    rows = [{
        "time_utc": w.time_utc.isoformat().replace("+00:00", "Z"),
        "observed_hz": float(o),
        "predicted_hz": float(p),
        "offset_hz": float(off),
        "residual_hz": float(r),
    } for w, o, p, off, r in zip(windows, observed, predicted, offsets,
                                 residuals)]
    return CandidateResult(satellite_id, "comparable",
                           lo_offset_hz=lo_offset, rms_residual_hz=rms,
                           rows=rows)


def decide(results: list[CandidateResult], rms_limit_hz: float,
           separation_hz: float) -> dict:
    """Apply the no-match / ambiguous / identified decision rules."""
    comparable = [r for r in results if r.status == "comparable"]
    comparable.sort(key=lambda r: (r.rms_residual_hz, r.satellite_id))
    if not comparable:
        return {"status": "no-match", "identified": None,
                "ambiguous_ids": [],
                "reason": "no comparable candidate remained"}
    best = comparable[0]
    if best.rms_residual_hz > rms_limit_hz:
        return {"status": "no-match", "identified": None,
                "ambiguous_ids": [],
                "reason": f"lowest residual RMS "
                          f"{best.rms_residual_hz:.3f} Hz exceeds the "
                          f"limit {rms_limit_hz:.3f} Hz"}
    tied = [r for r in comparable
            if r.rms_residual_hz - best.rms_residual_hz <= separation_hz]
    if len(tied) > 1:
        ids = sorted(r.satellite_id for r in tied)
        return {"status": "ambiguous", "identified": None,
                "ambiguous_ids": ids,
                "reason": f"candidates {', '.join(ids)} differ by no more "
                          f"than the separation {separation_hz:.3f} Hz; "
                          f"no unique identification claimed"}
    return {"status": "identified", "identified": best.satellite_id,
            "ambiguous_ids": [], "reason": None}
