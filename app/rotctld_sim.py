"""Local rotctld-compatible TCP rotator simulator.

Usage: .venv/bin/python -m app.rotctld_sim [--host 127.0.0.1] [--port 4533]

Implements newline-framed P (set position), p (read position) and S
(stop) with RPRT replies. The simulated axes slew toward the setpoint at
a fixed rate; S halts the motion immediately.
"""
from __future__ import annotations

import argparse
import socket
import socketserver
import threading
import time

SLEW_DPS = 5.0  # simulated axis speed, deg/s


class _Rotator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.az = 0.0
        self.el = 0.0
        self.target = (0.0, 0.0)
        self.moving = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_target(self, az: float, el: float) -> None:
        with self._lock:
            self.target = (az, el)
            self.moving = True

    def stop(self) -> None:
        with self._lock:
            self.moving = False

    def position(self) -> tuple[float, float]:
        with self._lock:
            return self.az, self.el

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            dt = now - last
            last = now
            with self._lock:
                if not self.moving:
                    continue
                taz, tel = self.target
                done = True
                for axis in ("az", "el"):
                    cur = getattr(self, axis)
                    tgt = taz if axis == "az" else tel
                    step = SLEW_DPS * dt
                    if abs(tgt - cur) <= step:
                        setattr(self, axis, tgt)
                    else:
                        setattr(self, axis,
                                cur + step if tgt > cur else cur - step)
                        done = False
                if done:
                    self.moving = False

    def close(self) -> None:
        self._stop.set()


ROTATOR = _Rotator()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            parts = line.decode("ascii", "replace").strip().split()
            if not parts:
                continue
            cmd = parts[0]
            if cmd == "P" and len(parts) == 3:
                try:
                    az, el = float(parts[1]), float(parts[2])
                except ValueError:
                    self.wfile.write(b"RPRT -1\n")
                    continue
                ROTATOR.set_target(az, el)
                self.wfile.write(b"RPRT 0\n")
            elif cmd == "p":
                az, el = ROTATOR.position()
                self.wfile.write(f"{az:.3f}\n{el:.3f}\n".encode())
            elif cmd == "S":
                ROTATOR.stop()
                self.wfile.write(b"RPRT 0\n")
            else:
                self.wfile.write(b"RPRT -1\n")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="rotctld simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4533)
    args = parser.parse_args()
    with _Server((args.host, args.port), _Handler) as srv:
        print(f"rotctld simulator on {args.host}:{args.port}", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    ROTATOR.close()


if __name__ == "__main__":
    main()

