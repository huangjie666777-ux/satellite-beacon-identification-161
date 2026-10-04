"""SigMF meta/data validation for single-channel cf32_le IQ recordings.

Only the strict subset needed here is accepted: one capture segment
starting at sample 0, no header/trailing bytes, datatype cf32_le.
The centre frequency is read from the capture's core:frequency, falling
back to the global core:frequency. Multi-channel recordings and
non-zero header/trailing byte counts are rejected.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_SAMPLES = 1 << 20          # 2^20 complex samples
MAX_DURATION_S = 60.0
MIN_SAMPLE_RATE_HZ = 1_000.0
MAX_SAMPLE_RATE_HZ = 200_000.0
MAX_META_BYTES = 1 << 20
MAX_DATA_BYTES = MAX_SAMPLES * 8
DATATYPE = "cf32_le"
BYTES_PER_SAMPLE = 8           # 2 x float32 little-endian


class SigmfError(ValueError):
    """Raised when SigMF metadata or sample data fail validation."""


@dataclass
class RecordingInfo:
    meta: dict
    sample_rate_hz: float
    center_frequency_hz: float
    start_time: datetime      # UTC recording start
    sample_count: int
    duration_s: float


def _finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _reject_unsupported(container: dict, where: str) -> None:
    num_ch = container.get("core:num_channels", 1)
    if (not isinstance(num_ch, int) or isinstance(num_ch, bool)
            or num_ch != 1):
        raise SigmfError(
            f"{where} core:num_channels must be 1, got {num_ch!r}: "
            "multi-channel recordings are not supported")
    for key in ("core:header_bytes", "core:trailing_bytes"):
        extra = container.get(key, 0)
        if (not isinstance(extra, int) or isinstance(extra, bool)
                or extra != 0):
            raise SigmfError(
                f"{where} {key} must be 0 or absent, got {extra!r}: "
                "additional header/trailing bytes are not supported")


def parse_start_time(value) -> datetime:
    if not isinstance(value, str):
        raise SigmfError("captures[0] core:datetime must be an ISO 8601 string")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SigmfError(
            f"captures[0] core:datetime is not valid ISO 8601: {value!r}"
        ) from exc
    if dt.tzinfo is None:
        raise SigmfError("captures[0] core:datetime must carry a UTC offset")
    return dt.astimezone(timezone.utc)


def validate(meta_raw: bytes, data_len: int) -> RecordingInfo:
    """Validate SigMF meta JSON and the raw data length; return summary."""
    if not meta_raw:
        raise SigmfError("empty SigMF metadata")
    if len(meta_raw) > MAX_META_BYTES:
        raise SigmfError("SigMF metadata exceeds 1 MiB")
    try:
        meta = json.loads(meta_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SigmfError(f"SigMF metadata is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise SigmfError("SigMF metadata root must be a JSON object")

    glob = meta.get("global")
    if not isinstance(glob, dict):
        raise SigmfError("SigMF metadata must contain a global object")

    datatype = glob.get("core:datatype")
    if datatype != DATATYPE:
        raise SigmfError(
            f"core:datatype must be {DATATYPE!r}, got {datatype!r}")

    _reject_unsupported(glob, "global")

    sample_rate = glob.get("core:sample_rate")
    if not _finite_number(sample_rate):
        raise SigmfError("global core:sample_rate must be a finite number")
    if not (MIN_SAMPLE_RATE_HZ <= sample_rate <= MAX_SAMPLE_RATE_HZ):
        raise SigmfError(
            f"core:sample_rate {sample_rate} Hz outside "
            f"[{MIN_SAMPLE_RATE_HZ}, {MAX_SAMPLE_RATE_HZ}] Hz")

    captures = meta.get("captures")
    if not isinstance(captures, list) or len(captures) != 1:
        raise SigmfError("exactly one capture segment is required")
    cap = captures[0]
    if not isinstance(cap, dict):
        raise SigmfError("captures[0] must be an object")
    _reject_unsupported(cap, "captures[0]")
    sample_start = cap.get("core:sample_start")
    if (not isinstance(sample_start, int) or isinstance(sample_start, bool)
            or sample_start != 0):
        raise SigmfError("captures[0] core:sample_start must be 0")
    if "core:datetime" not in cap:
        raise SigmfError("captures[0] must carry core:datetime (UTC start)")
    start_time = parse_start_time(cap["core:datetime"])

    # capture-level frequency wins; fall back to the global value
    if "core:frequency" in cap:
        center = cap["core:frequency"]
        where = "captures[0]"
    else:
        center = glob.get("core:frequency")
        where = "global"
    if not _finite_number(center) or center <= 0.0:
        raise SigmfError(f"{where} core:frequency must be a positive finite "
                         "number (Hz)")

    annotations = meta.get("annotations", [])
    if not isinstance(annotations, list):
        raise SigmfError("annotations must be a list")

    # no header/trailing bytes are allowed: the data file must be exactly
    # sample_count * 8 bytes of cf32_le
    if data_len == 0:
        raise SigmfError("empty recording: data file has no samples")
    if data_len > MAX_DATA_BYTES:
        raise SigmfError(f"data file exceeds {MAX_SAMPLES} complex samples")
    if data_len % BYTES_PER_SAMPLE != 0:
        raise SigmfError(
            f"data file is truncated: {data_len} bytes is not a multiple "
            f"of {BYTES_PER_SAMPLE} (cf32_le)")
    count = data_len // BYTES_PER_SAMPLE
    duration = count / sample_rate
    if duration > MAX_DURATION_S:
        raise SigmfError(
            f"recording duration {duration:.3f} s exceeds "
            f"{MAX_DURATION_S:.0f} s")
    return RecordingInfo(meta=meta, sample_rate_hz=float(sample_rate),
                         center_frequency_hz=float(center),
                         start_time=start_time, sample_count=count,
                         duration_s=duration)
