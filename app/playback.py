"""Exclusive playback controller.

Replays a mechanical track plan against a local rotctld TCP endpoint on a
monotonic-clock relative timeline. Only one playback may run at a time.
On timeout, disconnect or cancel it stops sending further commands,
best-effort sends S, releases the controller and keeps the real final
state for querying.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from .rotctl import RotctlClient, RotctlError

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

_POLL_S = 0.05


def build_timeline(plan: dict[str, Any]) -> list[tuple[float, float, float]]:
    """(t_rel_s, az_deg, el_deg) setpoints: preset ramp, tracking, homing."""
    cur = plan["current_position"]
    home = plan["home_position"]
    preset_s = float(plan["preset_seconds"])
    homing_s = float(plan["homing_seconds"])
    targets = plan["targets"]
    if not targets:
        raise ValueError("plan has no tracking targets")
    first = targets[0]
    last = targets[-1]

    def ramp(t0: float, dur: float, a0: float, e0: float,
             a1: float, e1: float) -> list[tuple[float, float, float]]:
        pts = []
        n = int(dur)
        for k in range(1, n + 1):
            frac = k / dur
            pts.append((t0 + k, a0 + (a1 - a0) * frac, e0 + (e1 - e0) * frac))
        if n < dur:  # fractional tail ends exactly at the segment end
            pts.append((t0 + dur, a1, e1))
        return pts

    timeline = [(0.0, float(cur["az_deg"]), float(cur["el_deg"]))]
    timeline += ramp(0.0, preset_s,
                     float(cur["az_deg"]), float(cur["el_deg"]),
                     float(first["az_deg"]), float(first["el_deg"]))
    for tgt in targets:
        timeline.append((preset_s + float(tgt["t_rel_s"]),
                         float(tgt["az_deg"]), float(tgt["el_deg"])))
    t_end = preset_s + float(last["t_rel_s"])
    timeline += ramp(t_end, homing_s,
                     float(last["az_deg"]), float(last["el_deg"]),
                     float(home["az_deg"]), float(home["el_deg"]))
    return timeline


class PlaybackController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {"state": STATE_IDLE, "detail": None,
                                        "progress": None, "last_position": None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _set(self, **kw: Any) -> None:
        with self._lock:
            self._status.update(kw)

    def submit(self, plan: dict[str, Any], host: str, port: int,
               tolerance_deg: float, timeout_s: float) -> None:
        # build the timeline synchronously so malformed plans (e.g. no
        # targets) are rejected before the controller is marked running
        timeline = build_timeline(plan)
        with self._lock:
            if self._status["state"] == STATE_RUNNING:
                raise RuntimeError("a playback is already running")
            self._cancel.clear()
            self._status = {"state": STATE_RUNNING, "detail": None,
                            "progress": {"sent": 0, "total": 0},
                            "last_position": None}
        self._thread = threading.Thread(
            target=self._run,
            args=(timeline, host, port, tolerance_deg, timeout_s),
            daemon=True)
        self._thread.start()

    def cancel(self) -> bool:
        with self._lock:
            running = self._status["state"] == STATE_RUNNING
        if running:
            self._cancel.set()
        return running

    def _run(self, timeline: list[tuple[float, float, float]], host: str, port: int,
             tolerance_deg: float, timeout_s: float) -> None:
        self._set(progress={"sent": 0, "total": len(timeline)})
        client: RotctlClient | None = None
        final_state = STATE_COMPLETED
        detail = None
        try:
            client = RotctlClient(host, port, timeout_s=timeout_s)
            # verify the actual position matches the plan's starting point
            az, el = client.get_position()
            self._set(last_position={"az_deg": az, "el_deg": el})
            start_az, start_el = timeline[0][1], timeline[0][2]
            if (abs(az - start_az) > tolerance_deg
                    or abs(el - start_el) > tolerance_deg):
                raise RotctlError(
                    f"actual position ({az:.3f}, {el:.3f}) differs from "
                    f"planned start ({start_az:.3f}, {start_el:.3f}) by "
                    f"more than {tolerance_deg} deg")
            t0 = time.monotonic()
            for idx, (t_rel, az, el) in enumerate(timeline[1:], start=1):
                while True:
                    if self._cancel.is_set():
                        raise _Cancelled()
                    remaining = t0 + t_rel - time.monotonic()
                    if remaining <= 0.0:
                        break
                    time.sleep(min(_POLL_S, remaining))
                client.set_position(az, el)
                self._set(progress={"sent": idx + 1, "total": len(timeline)},
                          last_position={"az_deg": az, "el_deg": el})
        except _Cancelled:
            final_state, detail = STATE_CANCELLED, "cancelled by request"
        except RotctlError as exc:
            final_state, detail = STATE_FAILED, str(exc)
        except Exception as exc:  # never leave the controller stuck running
            final_state, detail = STATE_FAILED, f"unexpected error: {exc}"
        finally:
            if client is not None:
                if final_state != STATE_COMPLETED:
                    try:
                        client.stop()  # best-effort S
                    except RotctlError:
                        pass
                client.close()
            self._set(state=final_state, detail=detail)


class _Cancelled(Exception):
    pass
