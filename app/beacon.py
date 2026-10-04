"""Unknown narrowband beacon identification.

A single stable unmodulated carrier is assumed. The recording is split
into non-overlapping 1024-point Hann windows; each window's spectral
peak is refined with a 3-point parabolic interpolation that handles the
FFT wrap boundary. Windows whose peak power does not exceed 10x the
median spectral power of the window are dropped. The remaining windows
must span at least 5 s.

For every candidate satellite the baseband shift of its downlink
(transmit) frequency is predicted at each accepted window centre using
the same SGP4 propagation and site geometry as the forecast. The
constant local-oscillator offset is estimated as mean(observed -
predicted); the de-biased residual RMS scores the candidate.

Decision (limits are positive finite Hz values from the request):
- candidates whose |bias| exceeds the bias limit are excluded;
- if no comparable candidate remains, or the best residual RMS exceeds
  the RMS limit, the status is "no-match";
- if the two best RMS values differ by no more than the separation, the
  status is "ambiguous" (no unique identification is claimed);
- otherwise the status is "identified".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .doppler import WINDOW_LEN, shift_nodes
from .passes import Site, find_passes
from .schemas import MAX_EPOCH_AGE
from .tle import TLEError, parse_tle

PEAK_TO_MEDIAN_MIN = 10.0
MIN_SPAN_S = 5.0


class BeaconError(ValueError):
    """Raised when the recording cannot yield a usable measurement."""


@dataclass
class WindowRow:
    index: int
    time_utc: datetime          # window centre
    t_rel_s: float              # seconds of the centre since recording start
    freq_hz: float              # parabolic-refined peak frequency (signed)
    peak_power: float
    median_power: float
    accepted: bool


def refined_peak_freq(power: np.ndarray, sample_rate_hz: float) -> float:
    """Parabolic-refined peak frequency (Hz, signed) of a power spectrum
    in FFT bin order; neighbour bins wrap around the FFT boundary.
    The parabola is fitted to the log power, which is a good match to
    the Hann mainlobe (sub-Hz error at 1024 points)."""
    n = power.shape[0]
    k = int(np.argmax(power))
    tiny = np.finfo(np.float64).tiny
    y0 = math.log(max(float(power[(k - 1) % n]), tiny))
    y1 = math.log(max(float(power[k]), tiny))
    y2 = math.log(max(float(power[(k + 1) % n]), tiny))
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0
    if denom < 0.0:
        delta = 0.5 * (y0 - y2) / denom
        delta = max(-0.5, min(0.5, delta))
    freq = ((k + delta) % n) * sample_rate_hz / n
    if freq >= sample_rate_hz / 2.0:
        freq -= sample_rate_hz
    return freq


def measure_windows(samples: np.ndarray, sample_rate_hz: float,
                    start_time: datetime,
                    window_len: int = WINDOW_LEN) -> list[WindowRow]:
    """Per non-overlapping Hann-window peak measurement with UTC stamp."""
    count = len(samples) // window_len
    if count == 0:
        return []
    window = np.hanning(window_len)
    rows: list[WindowRow] = []
    for k in range(count):
        seg = samples[k * window_len:(k + 1) * window_len]
        power = np.abs(np.fft.fft(seg * window)) ** 2
        kmax = int(np.argmax(power))
        peak = float(power[kmax])
        median = float(np.median(power))
        accepted = (median > 0.0
                    and peak > PEAK_TO_MEDIAN_MIN * median)
        t_rel = (k + 0.5) * window_len / sample_rate_hz
        rows.append(WindowRow(
            index=k,
            time_utc=start_time + timedelta(seconds=t_rel),
            t_rel_s=t_rel,
            freq_hz=refined_peak_freq(power, sample_rate_hz),
            peak_power=peak,
            median_power=median,
            accepted=accepted))
    return rows


def accepted_span_s(rows: list[WindowRow]) -> float:
    """Span between the first and last accepted window centres."""
    kept = [r for r in rows if r.accepted]
    if len(kept) < 2:
        return 0.0
    return kept[-1].t_rel_s - kept[0].t_rel_s


@dataclass
class CandidateOutcome:
    satellite_id: str
    status: str                       # "comparable" | "excluded"
    reason: str | None = None
    bias_hz: float | None = None
    rms_hz: float | None = None
    rows: list[dict] = field(default_factory=list)


def evaluate_candidate(sat_in, site: Site, t0: datetime, t1: datetime,
                       kept: list[WindowRow], sample_rate_hz: float,
                       center_hz: float) -> CandidateOutcome:
    """Score one candidate; exclusions carry a human-readable reason."""
    sat_id = sat_in.id
    tx_hz = sat_in.downlink_frequency_hz
    try:
        tle = parse_tle(sat_in.tle_line1, sat_in.tle_line2)
    except TLEError as exc:
        return CandidateOutcome(sat_id, "excluded",
                                reason=f"invalid TLE: {exc}")
    for edge in (t0, t1):
        age = edge - tle.epoch
        if age < -MAX_EPOCH_AGE or age > MAX_EPOCH_AGE:
            return CandidateOutcome(
                sat_id, "excluded",
                reason="recording time is more than 7 days from the TLE "
                       f"epoch {tle.epoch.isoformat()}")
    intervals = find_passes(tle.satrec, site, t0, t1)
    if not any(iv.start <= t0 and iv.end >= t1 for iv in intervals):
        return CandidateOutcome(
            sat_id, "excluded",
            reason="recording is not fully inside a single visibility "
                   "interval of this satellite/station")

    rel = np.array([r.t_rel_s for r in kept], dtype=np.float64)
    predicted = shift_nodes(tle.satrec, site, t0, rel, tx_hz, center_hz)
    nyquist = sample_rate_hz / 2.0
    worst = float(np.max(np.abs(predicted)))
    if worst >= nyquist:
        return CandidateOutcome(
            sat_id, "excluded",
            reason=f"predicted baseband shift reaches {worst:.1f} Hz, not "
                   f"below half the sample rate ({nyquist:.1f} Hz)")

    observed = np.array([r.freq_hz for r in kept], dtype=np.float64)
    bias = float(np.mean(observed - predicted))
    residuals = observed - predicted - bias
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    rows = [{"time_utc": r.time_utc.isoformat().replace("+00:00", "Z"),
             "observed_hz": float(o), "predicted_hz": float(p),
             "residual_hz": float(e)}
            for r, o, p, e in zip(kept, observed, predicted, residuals)]
    return CandidateOutcome(sat_id, "comparable", bias_hz=bias,
                            rms_hz=rms, rows=rows)


def decide(outcomes: list[CandidateOutcome], bias_limit_hz: float,
           rms_limit_hz: float, separation_hz: float):
    """Apply the bias/RMS/separation rules; return (status, ranking)."""
    for o in outcomes:
        if o.status == "comparable" and abs(o.bias_hz) > bias_limit_hz:
            o.status = "excluded"
            o.reason = (f"estimated LO bias {o.bias_hz:+.3f} Hz exceeds "
                        f"the limit {bias_limit_hz:.3f} Hz")
    comparable = sorted((o for o in outcomes if o.status == "comparable"),
                        key=lambda o: (o.rms_hz, o.satellite_id))
    if not comparable:
        status = "no-match"
    elif comparable[0].rms_hz > rms_limit_hz:
        status = "no-match"
    elif (len(comparable) >= 2
            and comparable[1].rms_hz - comparable[0].rms_hz
            <= separation_hz):
        status = "ambiguous"
    else:
        status = "identified"
    ranking = []
    for rank, o in enumerate(comparable, start=1):
        ranking.append({"rank": rank, "satellite_id": o.satellite_id,
                        "status": "comparable",
                        "bias_hz": o.bias_hz, "rms_hz": o.rms_hz,
                        "reason": None})
    for o in sorted((o for o in outcomes if o.status == "excluded"),
                    key=lambda o: o.satellite_id):
        ranking.append({"rank": None, "satellite_id": o.satellite_id,
                        "status": "excluded", "bias_hz": o.bias_hz,
                        "rms_hz": o.rms_hz, "reason": o.reason})
    return status, ranking
