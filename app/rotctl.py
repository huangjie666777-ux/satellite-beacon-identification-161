"""Minimal rotctld-compatible TCP client.

Newline-framed text protocol:
- "P <az> <el>"  set position, answered with "RPRT <code>" (0 == ok)
- "p"            read position, answered with two lines: azimuth, elevation
- "S"            stop, answered with "RPRT <code>"
Any non-zero RPRT code or malformed reply raises RotctlError.
"""
from __future__ import annotations

import socket


class RotctlError(RuntimeError):
    """Raised on protocol errors, RPRT failures, timeouts or disconnects."""


class RotctlClient:
    def __init__(self, host: str, port: int, timeout_s: float = 5.0):
        try:
            self._sock = socket.create_connection((host, port),
                                                  timeout=timeout_s)
        except OSError as exc:
            raise RotctlError(f"cannot connect to {host}:{port}: {exc}") from exc
        self._buf = b""

    def _readline(self) -> str:
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(4096)
            except (OSError, socket.timeout) as exc:
                raise RotctlError(f"read failed: {exc}") from exc
            if not chunk:
                raise RotctlError("connection closed by rotctld")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", "replace").strip()

    def _send(self, line: str) -> None:
        try:
            self._sock.sendall(line.encode("ascii") + b"\n")
        except OSError as exc:
            raise RotctlError(f"write failed: {exc}") from exc

    @staticmethod
    def _expect_ok(resp: str) -> None:
        parts = resp.split()
        if len(parts) != 2 or parts[0] != "RPRT":
            raise RotctlError(f"unexpected reply: {resp!r}")
        try:
            code = int(parts[1])
        except ValueError as exc:
            raise RotctlError(f"malformed RPRT reply: {resp!r}") from exc
        if code != 0:
            raise RotctlError(f"rotctld reported RPRT {code}")

    def set_position(self, az_deg: float, el_deg: float) -> None:
        self._send(f"P {az_deg:.3f} {el_deg:.3f}")
        self._expect_ok(self._readline())

    def get_position(self) -> tuple[float, float]:
        self._send("p")
        try:
            az = float(self._readline())
            el = float(self._readline())
        except ValueError as exc:
            raise RotctlError("malformed position reply") from exc
        return az, el

    def stop(self) -> None:
        self._send("S")
        self._expect_ok(self._readline())

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

