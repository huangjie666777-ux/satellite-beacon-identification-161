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

from app.beacon import (decide, measure_windows, refined_peak_freq,
                        CandidateOutcome)
from app.doppler import shift_node_times, shift_nodes
from app.main import app
from app.passes import HorizonMask, Site, find_passes
from app.sigmf import SigmfError
from app.sigmf import validate as sigmf_validate
from app.tle import parse_tle

client = TestClient(app)

FS = 48_000.0
DURATION_S = 12.0
TX_HZ = 145_800_000.0
CENTER_HZ = 145_798_500.0
LO_HZ = 180.0


def beacon_request():
    return json.loads(Path("examples/beacon_request.json").read_text())


def ambiguous_request():
    return json.loads(
        Path("examples/beacon_ambiguous_request.json").read_text())


def _recording(duration=DURATION_S, lo_hz=LO_HZ, noise=1e-3, carrier=True):
    """Deterministic in-memory beacon recording (ISS candidate)."""
    req = beacon_request()
    sat = next(s for s in req["satellites"] if s["id"] == "ISS")
    stn = req["stations"][0]
    tle = parse_tle(sat["tle_line1"], sat["tle_line2"])
    site = Site("BEIJING", stn["lat_deg"], stn["lon_deg"], stn["alt_m"],
                HorizonMask(stn["mask"]))
    start = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 4, 30, tzinfo=timezone.utc)
    iv = find_passes(tle.satrec, site, start, end)[0]
    t0 = (iv.start + timedelta(seconds=2.0)).replace(microsecond=0)
    node_times = shift_node_times(duration)
    shifts = shift_nodes(tle.satrec, site, t0, node_times, TX_HZ, CENTER_HZ)
    n = int(FS * duration)
    t = np.arange(n, dtype=np.float64) / FS
    rng = np.random.default_rng(11)
    noise_sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    noise_sig *= noise / np.sqrt(2.0)
    if carrier:
        inst = np.interp(t, node_times, shifts) + lo_hz
        phase = np.concatenate(
            ([0.0], np.cumsum(0.5 * (inst[:-1] + inst[1:]) / FS)))
        samples = 0.5 * np.exp(2j * np.pi * phase) + noise_sig
    else:
        samples = noise_sig
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


def post(meta, data_bytes, req=None, stn="BEIJING", bias=500.0, rms=50.0,
         sep=2.0, path="/api/beacon/identify"):
    files = {
        "forecast": (None, json.dumps(req or beacon_request())),
        "station_id": (None, stn),
        "bias_limit_hz": (None, str(bias)),
        "rms_limit_hz": (None, str(rms)),
        "separation_hz": (None, str(sep)),
        "meta": ("rec.sigmf-meta", json.dumps(meta), "application/json"),
        "data": ("rec.sigmf-data", data_bytes, "application/octet-stream"),
    }
    return client.post(path, files=files)


# ---- unit: spectral peak refinement -------------------------------------

def _tone(freq, n=1024, fs=FS):
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * freq * t)


def test_parabolic_peak_refinement():
    bin_hz = FS / 1024
    freq = 100 * bin_hz + 0.3 * bin_hz
    power = np.abs(np.fft.fft(_tone(freq) * np.hanning(1024))) ** 2
    assert refined_peak_freq(power, FS) == pytest.approx(freq, abs=1.0)


def test_parabolic_peak_wraps_fft_boundary():
    bin_hz = FS / 1024
    # negative frequency: peak bin near N-1, neighbour wraps to bin 0
    freq = -(1.4 * bin_hz)
    power = np.abs(np.fft.fft(_tone(freq) * np.hanning(1024))) ** 2
    assert refined_peak_freq(power, FS) == pytest.approx(freq, abs=1.0)
    # positive frequency just below the Nyquist bin
    freq = 511.4 * bin_hz
    power = np.abs(np.fft.fft(_tone(freq) * np.hanning(1024))) ** 2
    assert refined_peak_freq(power, FS) == pytest.approx(freq, abs=1.0)


def test_window_rejection_and_span():
    fs = 8_000.0
    t = np.arange(int(fs * 12)) / fs
    carrier = np.exp(2j * np.pi * 1000.0 * t)
    # first 32 windows are digital zeros: no spectral peak -> dropped
    samples = carrier.copy()
    samples[: 32 * 1024] = 0.0
    rows = measure_windows(samples.astype("<c8"), fs,
                           datetime(2024, 1, 1, tzinfo=timezone.utc))
    assert rows
    cut = 32
    assert all((not r.accepted) for r in rows if r.index < cut)
    assert all(r.accepted for r in rows if r.index >= cut)
    kept = [r for r in rows if r.accepted]
    assert kept[-1].t_rel_s - kept[0].t_rel_s >= 5.0
    assert rows[0].time_utc.isoformat().startswith("2024-01-01T00:00:00")


def test_decide_rules():
    def cand(sid, rms, bias=0.0):
        return CandidateOutcome(sid, "comparable", bias_hz=bias, rms_hz=rms)
    status, ranking = decide([cand("B", 1.0), cand("A", 1.0)], 10, 5, 0.5)
    assert status == "ambiguous"
    assert [r["satellite_id"] for r in ranking[:2]] == ["A", "B"]  # id order
    status, _ = decide([cand("A", 1.0), cand("B", 3.0)], 10, 5, 0.5)
    assert status == "identified"
    status, _ = decide([cand("A", 9.0)], 10, 5, 0.5)
    assert status == "no-match"          # best RMS over the limit
    status, ranking = decide([cand("A", 1.0, bias=50.0)], 10, 5, 0.5)
    assert status == "no-match"          # bias over the limit -> excluded
    assert ranking[0]["status"] == "excluded"
    assert "bias" in ranking[0]["reason"]


# ---- endpoint ------------------------------------------------------------

def test_identify_happy_path():
    meta, samples = _recording()
    r = post(meta, samples.tobytes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "identified"
    assert body["identified_satellite_id"] == "ISS"
    ranks = {e["satellite_id"]: e for e in body["ranking"]}
    assert ranks["ISS"]["rank"] == 1
    assert ranks["ISS"]["bias_hz"] == pytest.approx(LO_HZ, abs=0.5)
    assert ranks["ISS"]["rms_hz"] < 1.0
    assert ranks["ISS-DRIFT"]["rms_hz"] > ranks["ISS"]["rms_hz"] + 2.0
    assert body["windows"]["accepted_span_s"] >= 5.0


def test_identify_download_zip():
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), path="/api/beacon/identify/download")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(zf.namelist()) == [
        "observations.csv", "residuals/ISS-DRIFT.csv", "residuals/ISS.csv",
        "summary.json"]
    summary = json.loads(zf.read("summary.json"))
    assert summary["status"] == "identified"
    obs = zf.read("observations.csv").decode().splitlines()
    assert obs[0] == "window_index,time_utc,peak_freq_hz,peak_power," \
                     "median_power,accepted"
    assert len(obs) == 1 + summary["windows"]["total"]
    res = zf.read("residuals/ISS.csv").decode().splitlines()
    assert res[0] == "time_utc,observed_hz,predicted_hz,residual_hz"
    assert len(res) == 1 + summary["windows"]["accepted"]


def test_identify_ambiguous():
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), req=ambiguous_request())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ambiguous"
    assert body["identified_satellite_id"] is None
    comparable = [e for e in body["ranking"] if e["status"] == "comparable"]
    assert [e["satellite_id"] for e in comparable] == ["SAT-A", "SAT-B"]


def test_identify_no_match_rms_limit():
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), rms=1e-6)
    assert r.status_code == 200
    assert r.json()["status"] == "no-match"


def test_identify_bias_limit_excludes():
    meta, samples = _recording()
    r = post(meta, samples.tobytes(), bias=10.0)  # LO offset is 180 Hz
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no-match"
    assert all(e["status"] == "excluded" for e in body["ranking"])
    assert all("bias" in e["reason"] for e in body["ranking"])


def test_identify_candidate_excluded_nyquist():
    meta, samples = _recording()
    req = beacon_request()
    # transmit frequency far from the recording centre -> shift > fs/2
    req["satellites"][1]["downlink_frequency_hz"] = 145_900_000.0
    r = post(meta, samples.tobytes(), req=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "identified"
    drift = next(e for e in body["ranking"]
                 if e["satellite_id"] == "ISS-DRIFT")
    assert drift["status"] == "excluded"
    assert "half the sample rate" in drift["reason"]


def test_identify_candidate_excluded_visibility():
    meta, samples = _recording()
    req = beacon_request()
    # a TLE whose pass does not cover the recording window
    req["satellites"][1] = {
        "id": "NOAA15",
        "tle_line1": "1 25338U 98030A   24001.50000000  .00001107  00000-0  26088-3 0  9999",
        "tle_line2": "2 25338  98.7000  33.7847 0010704  55.5500 304.6776 14.26198444336997",
        "downlink_frequency_hz": 137_620_000.0,
    }
    r = post(meta, samples.tobytes(), req=req)
    assert r.status_code == 200, r.text
    body = r.json()
    noaa = next(e for e in body["ranking"]
                if e["satellite_id"] == "NOAA15")
    assert noaa["status"] == "excluded"
    assert "visibility" in noaa["reason"]


def test_identify_rejects_noise_only_and_short():
    meta, samples = _recording(duration=2.0)  # span below 5 s
    r = post(meta, samples.tobytes())
    assert r.status_code == 422
    assert "span" in r.json()["detail"]
    # digital zeros: no window carries a spectral peak
    meta, samples = _recording()
    r = post(meta, np.zeros_like(samples).tobytes())
    assert r.status_code == 422
    assert "span" in r.json()["detail"]


def test_identify_rejects_bad_thresholds_and_ids():
    meta, samples = _recording()
    data = samples.tobytes()
    assert post(meta, data, bias=0.0).status_code == 422
    assert post(meta, data, rms=-1.0).status_code == 422
    assert post(meta, data, sep="inf").status_code == 422
    assert post(meta, data, sep="abc").status_code == 422
    assert post(meta, data, stn="NOPE").status_code == 422


# ---- SigMF subset rules ---------------------------------------------------

def test_sigmf_capture_frequency_and_fallback():
    meta, samples = _recording()
    data = samples.tobytes()
    # capture-level frequency wins
    m = copy.deepcopy(meta)
    m["global"]["core:frequency"] = 100_000_000.0
    info = sigmf_validate(json.dumps(m).encode(), len(data))
    assert info.center_frequency_hz == CENTER_HZ
    # global fallback when the capture has none
    m = copy.deepcopy(meta)
    del m["captures"][0]["core:frequency"]
    m["global"]["core:frequency"] = 100_000_000.0
    info = sigmf_validate(json.dumps(m).encode(), len(data))
    assert info.center_frequency_hz == 100_000_000.0
    # neither -> rejected
    m = copy.deepcopy(meta)
    del m["captures"][0]["core:frequency"]
    with pytest.raises(SigmfError):
        sigmf_validate(json.dumps(m).encode(), len(data))


def test_sigmf_rejects_multichannel_and_extra_bytes():
    meta, samples = _recording()
    data = samples.tobytes()
    for mutate in (
        lambda m: m["global"].update(**{"core:num_channels": 2}),
        lambda m: m["captures"][0].update(**{"core:num_channels": 2}),
        lambda m: m["global"].update(**{"core:header_bytes": 16}),
        lambda m: m["captures"][0].update(**{"core:trailing_bytes": 8}),
    ):
        m = copy.deepcopy(meta)
        mutate(m)
        with pytest.raises(SigmfError):
            sigmf_validate(json.dumps(m).encode(), len(data))
    # explicit single channel / zero extra bytes stay valid
    m = copy.deepcopy(meta)
    m["global"]["core:num_channels"] = 1
    m["global"]["core:header_bytes"] = 0
    sigmf_validate(json.dumps(m).encode(), len(data))
