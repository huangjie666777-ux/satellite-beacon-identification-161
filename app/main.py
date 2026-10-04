"""FastAPI entrypoint: offline satellite pass forecast service."""
from __future__ import annotations

import copy
import json
from datetime import timedelta

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .delivery import (build_iq_zip, build_zip, interval_csv_rows,
                       interval_to_dict, render_csv)
from .doppler import WINDOW_LEN, correct, shift_node_times, shift_nodes
from .doppler import window_diagnostics
from .passes import HorizonMask, PropagationError, Site, find_passes
from .playback import PlaybackController
from .schemas import (MAX_EPOCH_AGE, ForecastRequest, ForecastResponse,
                      IntervalOut, MechTargetOut, PlaybackRequest,
                      TrackPlanRequest, TrackPlanResponse)
from .sigmf import MAX_DATA_BYTES, MAX_META_BYTES, SigmfError
from .sigmf import validate as sigmf_validate
from .tle import TLEError, parse_tle
from .tracker import AxisLimits, PlanError, plan_track

app = FastAPI(title="Offline Pass Forecast", version="1.0.0")
playback = PlaybackController()

NOTES = [
    "Positions: SGP4/SDP4 (WGS72) in TEME, rotated to ECEF with GMST; "
    "UTC approximates UT1; no polar motion, refraction or light-time.",
    "Site coordinates are WGS84 geodetic; azimuth is clockwise from "
    "true north.",
    "Visibility requires elevation strictly above the interpolated "
    "horizon mask; tangential touches are not valid intervals.",
    "Search grid is 1 s with crossings bisected to 0.1 s; intervals "
    "shorter than ~1 s may be missed.",
    "Doppler shift: negative means the received frequency is below "
    "nominal (range increasing).",
]


def _prepare(req: ForecastRequest):
    try:
        sats = []
        for s in req.satellites:
            tle = parse_tle(s.tle_line1, s.tle_line2)
            age_start = req.window.start - tle.epoch
            age_end = req.window.end - tle.epoch
            if (age_start < -MAX_EPOCH_AGE or age_start > MAX_EPOCH_AGE
                    or age_end < -MAX_EPOCH_AGE or age_end > MAX_EPOCH_AGE):
                raise TLEError(
                    f"satellite {s.id}: query window is more than 7 days "
                    f"from the TLE epoch {tle.epoch.isoformat()}")
            sats.append((s, tle))
    except TLEError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sites = [Site(station_id=st.id, lat_deg=st.lat_deg, lon_deg=st.lon_deg,
                  alt_m=st.alt_m, mask=HorizonMask(st.mask))
             for st in req.stations]
    return sats, sites


def _compute(req: ForecastRequest):
    sats, sites = _prepare(req)
    results = []  # (sat_in, site, interval)
    try:
        for sat_in, tle in sats:
            for site in sites:
                for iv in find_passes(tle.satrec, site,
                                      req.window.start, req.window.end):
                    results.append((sat_in, tle, site, iv))
    except PropagationError as exc:
        # any propagation failure fails the whole request
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    results.sort(key=lambda r: (r[3].start, r[0].id))
    return results


@app.post("/api/passes", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    results = _compute(req)
    intervals = [IntervalOut(**interval_to_dict(s.id, st.station_id, iv))
                 for s, _t, st, iv in results]
    return ForecastResponse(window=req.window, interval_count=len(intervals),
                            intervals=intervals, notes=NOTES)


@app.post("/api/passes/download")
def forecast_download(req: ForecastRequest):
    results = _compute(req)
    summary = {
        "window": {"start": req.window.start.isoformat().replace("+00:00", "Z"),
                   "end": req.window.end.isoformat().replace("+00:00", "Z")},
        "interval_count": len(results),
        "intervals": [interval_to_dict(s.id, st.station_id, iv)
                      for s, _t, st, iv in results],
        "units": {"azimuth": "deg", "elevation": "deg", "range": "km",
                  "range_rate": "km/s", "doppler_shift": "Hz"},
        "notes": NOTES,
    }
    csv_files = {}
    for idx, (s, tle, st, iv) in enumerate(results):
        rows = interval_csv_rows(tle.satrec, st, iv, s.downlink_frequency_hz)
        csv_files[f"{s.id}_{st.station_id}_{idx:03d}.csv"] = render_csv(rows)
    payload = build_zip(summary, csv_files)
    return Response(
        content=payload, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="passes.zip"'})


@app.get("/api/health")
def health():
    return {"status": "ok"}


TRACK_NOTES = [
    "Mechanical azimuth may leave [0, 360) via +360*k unwrapping; no "
    "over-the-top elevation flip is used.",
    "Targets are sampled every 1 s and include both interval endpoints; "
    "t_rel_s is relative to the first tracking target.",
    "The path current -> preset -> tracking -> homing is chosen to "
    "minimize total azimuth travel; ties use the lexicographically "
    "smallest mechanical azimuth sequence.",
    "A single tracked segment must not exceed 30 minutes; infeasible "
    "intervals are rejected as a whole.",
]


@app.post("/api/track/plan", response_model=TrackPlanResponse)
def track_plan(req: TrackPlanRequest):
    results = _compute(req.forecast)
    if req.interval_index >= len(results):
        raise HTTPException(
            status_code=422,
            detail=f"interval_index {req.interval_index} out of range: "
                   f"{len(results)} interval(s) in window")
    sat, tle, site, iv = results[req.interval_index]
    limits = AxisLimits(
        az_min_deg=req.az_min_deg, az_max_deg=req.az_max_deg,
        el_min_deg=req.el_min_deg, el_max_deg=req.el_max_deg,
        max_az_rate_dps=req.max_az_rate_dps,
        max_el_rate_dps=req.max_el_rate_dps)
    try:
        plan = plan_track(
            tle.satrec, site, iv.start, iv.end, limits,
            current=(req.current_position.az_deg, req.current_position.el_deg),
            home=(req.home_position.az_deg, req.home_position.el_deg),
            preset_s=req.preset_seconds, homing_s=req.homing_seconds)
    except PlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrackPlanResponse(
        satellite_id=sat.id, station_id=site.station_id,
        interval_index=req.interval_index,
        interval_start=iv.start, interval_end=iv.end,
        preset_seconds=req.preset_seconds,
        homing_seconds=req.homing_seconds,
        current_position=req.current_position,
        home_position=req.home_position,
        targets=[MechTargetOut(t_rel_s=round(t.t_rel_s, 3),
                               az_deg=round(t.az_deg, 3),
                               el_deg=round(t.el_deg, 3))
                 for t in plan.targets],
        total_az_travel_deg=round(plan.total_az_travel_deg, 3),
        notes=TRACK_NOTES)


@app.post("/api/playback")
def playback_start(req: PlaybackRequest):
    try:
        playback.submit(req.plan.model_dump(), req.host, req.port,
                        req.position_tolerance_deg, req.response_timeout_s)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return playback.status()


@app.get("/api/playback")
def playback_status():
    return playback.status()


@app.post("/api/playback/cancel")
def playback_cancel():
    cancelled = playback.cancel()
    return {"cancel_requested": cancelled, **playback.status()}


IQ_NOTES = [
    "Recording must lie entirely inside a single visibility interval of "
    "the chosen satellite/station pair.",
    "Baseband shift = (f_tx - f_center) - f_tx * range_rate / c; "
    "range rate is the ECEF radial velocity, positive when receding.",
    "Shift nodes are computed on a 1 s grid including the recording end "
    "and linearly interpolated; correction integrates the shift from "
    "phase 0 at the recording start and multiplies by exp(-j*phi).",
    "No resampling and no normalisation: sample count, sample rate and "
    "amplitude are preserved.",
    "Diagnostics use non-overlapping 1024-point Hann windows; recordings "
    "shorter than one window have no per-window diagnostics.",
]


def _parse_tx_frequency(raw: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422,
                            detail="tx_frequency_hz must be a number") from exc
    if not (value > 0.0) or value != value or value in (float("inf"), float("-inf")):
        raise HTTPException(
            status_code=422,
            detail="tx_frequency_hz must be a positive finite number (Hz)")
    return value


@app.post("/api/doppler/correct")
async def doppler_correct(forecast: str = Form(...),
                          satellite_id: str = Form(...),
                          station_id: str = Form(...),
                          tx_frequency_hz: str = Form(...),
                          meta: UploadFile = File(...),
                          data: UploadFile = File(...)):
    tx_hz = _parse_tx_frequency(tx_frequency_hz)
    try:
        req = ForecastRequest.model_validate(json.loads(forecast))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422,
                            detail=f"invalid forecast request: {exc}") from exc
    sat_in = next((s for s in req.satellites if s.id == satellite_id), None)
    if sat_in is None:
        raise HTTPException(status_code=422,
                            detail=f"unknown satellite id {satellite_id!r}")
    stn_in = next((s for s in req.stations if s.id == station_id), None)
    if stn_in is None:
        raise HTTPException(status_code=422,
                            detail=f"unknown station id {station_id!r}")

    meta_raw = await meta.read(MAX_META_BYTES + 1)
    data_raw = await data.read(MAX_DATA_BYTES + 1)
    try:
        info = sigmf_validate(meta_raw, len(data_raw))
    except SigmfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    samples = np.frombuffer(data_raw, dtype="<c8").copy()
    if not (np.isfinite(samples.real).all()
            and np.isfinite(samples.imag).all()):
        raise HTTPException(status_code=422,
                            detail="recording contains non-finite samples")

    try:
        tle = parse_tle(sat_in.tle_line1, sat_in.tle_line2)
    except TLEError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    t0 = info.start_time
    t1 = t0 + timedelta(seconds=info.duration_s)
    for edge in (t0, t1):
        age = edge - tle.epoch
        if age < -MAX_EPOCH_AGE or age > MAX_EPOCH_AGE:
            raise HTTPException(
                status_code=422,
                detail="recording time is more than 7 days from the TLE "
                       f"epoch {tle.epoch.isoformat()}")
    site = Site(station_id=stn_in.id, lat_deg=stn_in.lat_deg,
                lon_deg=stn_in.lon_deg, alt_m=stn_in.alt_m,
                mask=HorizonMask(stn_in.mask))
    try:
        intervals = find_passes(tle.satrec, site, t0, t1)
    except PropagationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not any(iv.start <= t0 and iv.end >= t1 for iv in intervals):
        raise HTTPException(
            status_code=422,
            detail="recording is not fully inside a single visibility "
                   "interval of the chosen satellite/station")

    node_times = shift_node_times(info.duration_s)
    try:
        node_shifts = shift_nodes(tle.satrec, site, t0, node_times,
                                  tx_hz, info.center_frequency_hz)
    except PropagationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    nyquist = info.sample_rate_hz / 2.0
    worst = float(np.max(np.abs(node_shifts)))
    if worst >= nyquist:
        raise HTTPException(
            status_code=422,
            detail=f"baseband shift reaches {worst:.1f} Hz, not below "
                   f"half the sample rate ({nyquist:.1f} Hz)")

    corrected = correct(samples, info.sample_rate_hz, node_times, node_shifts)
    windows = window_diagnostics(samples, corrected, info.sample_rate_hz)

    out_meta = copy.deepcopy(info.meta)
    out_meta["global"]["core:frequency"] = tx_hz
    diagnostics = {
        "satellite_id": satellite_id,
        "station_id": station_id,
        "tx_frequency_hz": tx_hz,
        "center_frequency_hz": info.center_frequency_hz,
        "sample_rate_hz": info.sample_rate_hz,
        "sample_count": info.sample_count,
        "start_time_utc": t0.isoformat().replace("+00:00", "Z"),
        "duration_s": info.duration_s,
        "shift_nodes": [{"t_s": float(t), "shift_hz": float(s)}
                        for t, s in zip(node_times, node_shifts)],
        "window": {"length": WINDOW_LEN, "shape": "hann",
                   "overlap": 0, "count": len(windows)},
        "windows": windows,
        "notes": IQ_NOTES,
    }
    payload = build_iq_zip(out_meta, corrected.tobytes(), diagnostics)
    return Response(
        content=payload, media_type="application/zip",
        headers={"Content-Disposition":
                 'attachment; filename="doppler_corrected.zip"'})
