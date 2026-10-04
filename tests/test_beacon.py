"""Tests for the unknown narrowband beacon identification endpoint."""
import copy
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.beacon import measure_windows, valid_windows, BeaconError
from app.doppler import shift_node_times, shift_nodes
from app.main import app
from app.passes import HorizonMask, Site, find_passes
from app.tle import parse_tle

client = TestClient(app)

FS = 48_000.0
DURATION_S = 12.0
TX_HZ = 145_800_000.0
CENTER_HZ = 145_798_500.0
LO_HZ = 250.0


def beacon_request():
    return json.loads(Path("examples/beacon_request.json").read_text())


def _recording(duration=DURATION_S, lo_hz=LO_HZ, amplitude=0.5,
               pure_noise=False):
    req = beacon_request()
    sat = next(s for s in req["satellites"] if s["id"] == "ISS")
    stn = req["stations"][0]
    tle = parse_tle(sat["tle_line1"], sat["tle_line2"])
    site = Site(station_id="BEIJING", lat_deg=stn["lat_deg"],
                lon_deg=stn["lon_deg"], alt_m=stn["alt_m"],
                mask=HorizonMask(stn["mask"]))
    start = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 4, 30, tzinfo=timezone.utc)
    iv = find_passes(tle.satrec, site, start, end)[0]
    t0 = (iv.start + timedelta(seconds=2.0)).replace(microsecond=0)
    node_times = shift_node_times(duration)
    shifts = shift_nodes(tle.satrec, site, t0, node_times, TX_HZ, CENTER_HZ)
    n = int(FS * duration)
    t = np.arange(n, dtype=np.float64) / FS
    rng = np.random.default_rng(11)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3
    if pure_noise:
        samples = noise
    else:
        inst = np.interp(t, node_times, shifts) + lo_hz
        phase = np.concatenate(
            ([0.0], np.cumsum(0.5 * (inst[:-1] + inst[1:]) / FS)))
        samples = amplitude * np.exp(2j * np.pi * phase) + noise
    samples = samples.astype("<c8")
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": FS,
                   "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0,
                      "core:datetime": t0.isoformat().replace("+00:00", "Z"),
                      "core:frequency": CENTER_HZ}],
        "annotations": [],
    }
    return meta, samples


def post(meta, data_bytes, req=None, stn="BEIJING", lo_limit=500.0,
         rms_limit=1.0, separation=0.5, path="/api/beacon/identify"):
    files = {
        "forecast": (None, json.dumps(req or beacon_request())),
        "station_id": (None, stn),
        "lo_offset_limit_hz": (None, str(lo_limit)),
        "rms_limit_hz": (None, str(rms_limit)),
        "separation_hz": (None, str(separation)),
        "meta": ("rec.sigmf-meta", json.dumps(meta), "application/json"),
        "data": ("rec.sigmf-data", data_bytes, "application/octet-stream"),
    }
    return client.post(path, files=files)


def test_identified_single_candidate():
    meta, samples = _recording()
    req = beacon_request()
    req["satellites"] = req["satellites"][:1]  # only ISS
    r = post(meta, samples.tobytes(), req=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "identified"
    assert body["identified"] == "ISS"
    assert body["ambiguous_ids"] == []
    entry = body["ranking"][0]
    assert entry["satellite_id"] == "ISS"
    assert entry["status"] == "comparable"
    assert abs(entry["lo_offset_hz"] - LO_HZ) < 1.0
    assert entry["rms_residual_hz"] < 1.0
    assert body["windows"]["valid_span_s"] >= 5.0


def test_ambiguous_tie_listed_by_id():
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), separation=0.5)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ambiguous"
    assert body["identified"] is None
    assert body["ambiguous_ids"] == ["ISS", "ISS_ALT"]  # sorted by id
    comparable = [e for e in body["ranking"] if e["status"] == "comparable"]
    assert len(comparable) == 2
    diff = abs(comparable[0]["rms_residual_hz"]
               - comparable[1]["rms_residual_hz"])
    assert diff <= 0.5


def test_ambiguity_resolved_with_zero_separation():
    meta, samples = _recording()
    req = beacon_request()
    req["satellites"] = req["satellites"][:1]  # drop the near-twin
    r = post(meta, samples.tobytes(), req=req, separation=1e-9)
    body = r.json()
    assert body["status"] == "identified"
    assert body["identified"] == "ISS"


def test_lo_offset_limit_excludes():
    meta, samples = _recording()
    # true LO is 250 Hz; ISS_ALT (downlink 300 Hz higher) estimates -50 Hz,
    # so a 40 Hz limit excludes both candidates
    r = post(meta, samples.tobytes(), lo_limit=40.0)
    body = r.json()
    assert body["status"] == "no-match"
    assert all(e["status"] == "excluded" for e in body["ranking"])
    assert "LO offset" in body["ranking"][0]["reason"]


def test_lo_offset_limit_shifts_identification():
    # with a 100 Hz limit only ISS_ALT (estimated LO -50 Hz) stays
    # comparable, so it is the one identified
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), lo_limit=100.0)
    body = r.json()
    assert body["status"] == "identified"
    assert body["identified"] == "ISS_ALT"


def test_rms_limit_no_match():
    meta, samples = _recording()
    # residual RMS is ~0.26 Hz; a 0.1 Hz limit must produce no-match
    r = post(meta, samples.tobytes(), rms_limit=0.1)
    body = r.json()
    assert body["status"] == "no-match"
    assert "RMS" in body["reason"]


def test_excluded_candidate_outside_visibility():
    meta, samples = _recording()
    req = beacon_request()
    # shift the recording to a time with no ISS pass
    meta["captures"][0]["core:datetime"] = "2024-01-01T02:00:00Z"
    r = post(meta, samples.tobytes(), req=req)
    body = r.json()
    assert body["status"] == "no-match"
    assert all("visibility" in e["reason"] for e in body["ranking"])


def test_pure_noise_no_match():
    # white noise has no stable carrier: windows that survive the
    # peak-to-median test yield random frequencies, so the estimated LO
    # offset lands outside the limit and every candidate is excluded
    meta, samples = _recording(pure_noise=True)
    r = post(meta, samples.tobytes())
    assert r.status_code == 200
    assert r.json()["status"] == "no-match"


def test_no_carrier_windows_rejected():
    # an all-zero recording fails the peak-to-median test in every window
    meta, samples = _recording()
    r = post(meta, np.zeros_like(samples).tobytes())
    assert r.status_code == 422
    assert "carrier" in r.json()["detail"]


def test_short_recording_rejected():
    meta, samples = _recording(duration=3.0)
    r = post(meta, samples.tobytes())
    assert r.status_code == 422
    assert "5" in r.json()["detail"]


def test_reject_bad_limits():
    meta, samples = _recording()
    data = samples.tobytes()
    assert post(meta, data, lo_limit=0.0).status_code == 422
    assert post(meta, data, rms_limit=-1.0).status_code == 422
    assert post(meta, data, separation="inf").status_code == 422
    assert post(meta, data, separation="abc").status_code == 422


def test_reject_multi_channel_and_extra_bytes():
    meta, samples = _recording()
    data = samples.tobytes()
    m = copy.deepcopy(meta)
    m["global"]["core:num_channels"] = 2
    assert post(m, data).status_code == 422
    m = copy.deepcopy(meta)
    m["captures"][0]["core:offset"] = 16
    assert post(m, data).status_code == 422
    m = copy.deepcopy(meta)
    m["global"]["core:header_bytes"] = 32
    assert post(m, data).status_code == 422


def test_capture_frequency_precedence_and_global_fallback():
    meta, samples = _recording()
    data = samples.tobytes()
    # capture-level frequency wins over the global one
    m = copy.deepcopy(meta)
    m["global"]["core:frequency"] = 1.0  # nonsense, must be ignored
    assert post(m, data).status_code == 200
    # global frequency is used when the capture has none
    m = copy.deepcopy(meta)
    del m["captures"][0]["core:frequency"]
    m["global"]["core:frequency"] = CENTER_HZ
    assert post(m, data).status_code == 200
    # neither -> rejected
    m = copy.deepcopy(meta)
    del m["captures"][0]["core:frequency"]
    assert post(m, data).status_code == 422


def test_download_zip_contents():
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), path="/api/beacon/identify/download")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert "summary.json" in names
    assert "observations.csv" in names
    assert "candidates/ISS.csv" in names
    assert "candidates/ISS_ALT.csv" in names
    summary = json.loads(zf.read("summary.json"))
    assert summary["status"] == "ambiguous"
    obs = zf.read("observations.csv").decode().splitlines()
    assert obs[0].startswith("index,time_utc")
    assert len(obs) == 1 + summary["windows"]["total"]
    res = zf.read("candidates/ISS.csv").decode().splitlines()
    assert res[0] == "time_utc,observed_hz,predicted_hz,offset_hz,residual_hz"
    assert len(res) == 1 + summary["windows"]["valid"]


def test_parabolic_peak_wraps_fft_boundary():
    # carrier at a negative frequency between bins: refinement must land
    # within a fraction of a bin and handle the bin-0/N-1 neighbourhood
    fs = 48000.0
    n = 1024 * 8
    t = np.arange(n) / fs
    for freq in (-1200.37, -23.4, 0.0, 23437.13):
        samples = np.exp(2j * np.pi * freq * t).astype("<c8")
        ms = measure_windows(samples, fs,
                             datetime(2024, 1, 1, tzinfo=timezone.utc))
        for m in ms:
            assert m.freq_hz == pytest.approx(freq, abs=1.0)
            assert m.accepted


def test_valid_windows_span_rule():
    fs = 48000.0
    n = 1024 * 50  # ~1.07 s of carrier
    t = np.arange(n) / fs
    samples = np.exp(2j * np.pi * 1000.0 * t).astype("<c8")
    ms = measure_windows(samples, fs,
                         datetime(2024, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(BeaconError):
        valid_windows(ms)
