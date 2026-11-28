"""Decoupled USB CDC transport for STM32H7 500 Hz observations and 50 Hz RL PD commands.

Reuses ``telemetry_capture.FrameStream``, ``LossStats``, and ``build_rl_command`` so wire
framing and CRC16 match the STM32H7 firmware byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Callable, Protocol

HOST_DIR = Path(__file__).resolve().parent
DEPLOY_TOOLS_DIR = HOST_DIR.parent / "wheel_leg_deploy" / "Tools"
if str(DEPLOY_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_TOOLS_DIR))

from telemetry_capture import (  # noqa: E402
    FrameStream,
    LossStats,
    Observation,
    RL_FLAG_ZERO_TORQUE,
    UINT32_MASK,
    build_rl_command,
)


class ByteTransport(Protocol):
    @property
    def in_waiting(self) -> int: ...

    def read(self, size: int) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


class SerialPortTransport:
    """Thin wrapper around pyserial (lazy-imported so offline tests need no Win32 serial)."""

    def __init__(self, port: str, baud: int = 921600, *, timeout: float = 0.005, write_timeout: float = 0.05):
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyserial is required for hardware USB CDC: pip install pyserial") from exc
        self._serial_exc = serial.SerialException
        self._port = serial.Serial(
            port,
            baud,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=timeout,
            write_timeout=write_timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        try:
            self._port.reset_input_buffer()
            self._port.reset_output_buffer()
        except Exception:
            pass

    @property
    def in_waiting(self) -> int:
        return int(self._port.in_waiting)

    def read(self, size: int) -> bytes:
        return self._port.read(size)

    def write(self, data: bytes) -> int:
        written = self._port.write(data)
        return int(written) if written is not None else len(data)

    def close(self) -> None:
        self._port.close()


@dataclass(frozen=True)
class LinkStatsSnapshot:
    rx_frames: int
    lost_frames: int
    loss_percent: float
    restarts: int
    duplicates: int
    crc_errors: int
    length_errors: int
    discarded_bytes: int
    raw_rx_bytes: int
    tx_frames: int
    tx_errors: int
    connected: bool
    last_error: str | None


class UsbLink:
    """High-rate background USB CDC receiver + thread-safe RL PD command sender."""

    def __init__(
        self,
        transport: ByteTransport,
        *,
        ring_capacity: int = 8192,
        on_raw_bytes: Callable[[bytes], None] | None = None,
        on_observation: Callable[[Observation, int], None] | None = None,
    ):
        self._transport = transport
        self._stream = FrameStream(capacity=ring_capacity)
        self._loss = LossStats()
        self._on_raw_bytes = on_raw_bytes
        self._on_observation = on_observation

        self._rx_lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self._latest_obs: Observation | None = None
        self._latest_lost_before: int = 0
        self._raw_rx_bytes = 0
        self._tx_seq = 0
        self._tx_frames = 0
        self._tx_errors = 0
        self._start_ns = time.monotonic_ns()
        self._connected = True
        self._last_error: str | None = None
        self._stop_event = threading.Event()
        self._rx_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._rx_thread is not None and self._rx_thread.is_alive():
            return
        self._stop_event.clear()
        self._start_ns = time.monotonic_ns()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="usb-cdc-rx", daemon=True)
        self._rx_thread.start()

    def stop(self, *, send_zero_burst: bool = True) -> None:
        if send_zero_burst and self._connected:
            self.send_zero_burst(count=5, interval_s=0.004)
        self._stop_event.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
        try:
            self._transport.close()
        except Exception:
            pass
        self._connected = False

    def _rx_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                waiting = self._transport.in_waiting
                count = min(max(waiting, 1), 4096)
                chunk = self._transport.read(count)
                rx_ns = time.monotonic_ns()
            except Exception as exc:
                self._connected = False
                self._last_error = f"RX error: {exc}"
                return
            if not chunk:
                continue
            self._raw_rx_bytes += len(chunk)
            if self._on_raw_bytes is not None:
                self._on_raw_bytes(chunk)
            observations = self._stream.feed_received(chunk, rx_ns)
            for obs in observations:
                lost_before = self._loss.record(obs.seq, obs.timestamp_ms)
                if lost_before is None:
                    continue
                with self._rx_lock:
                    self._latest_obs = obs
                    self._latest_lost_before = lost_before
                if self._on_observation is not None:
                    self._on_observation(obs, lost_before)

    def get_latest_observation(self) -> tuple[Observation | None, int]:
        with self._rx_lock:
            return self._latest_obs, self._latest_lost_before

    def send_command(
        self,
        *,
        torque,
        kp,
        kd,
        q_ref,
        v_ref,
        source_seq: int,
        source_timestamp_ms: int,
        rl_flags: int,
    ) -> tuple[int, int, bool]:
        """Serialize and write one 140-byte atomic RL PD command frame.

        Returns ``(tx_seq, tx_timestamp_ms, ok)``.
        """
        with self._tx_lock:
            seq = self._tx_seq & UINT32_MASK
            ts_ms = int((time.monotonic_ns() - self._start_ns) // 1_000_000) & UINT32_MASK
            packet = build_rl_command(
                seq=seq,
                timestamp_ms=ts_ms,
                torque=torque,
                kp=kp,
                kd=kd,
                q_ref=q_ref,
                v_ref=v_ref,
                source_seq=source_seq,
                source_timestamp_ms=source_timestamp_ms,
                rl_flags=rl_flags,
            )
            try:
                written = self._transport.write(packet)
                if written != len(packet):
                    raise OSError(f"short USB write: {written}/{len(packet)}")
                self._tx_seq = (self._tx_seq + 1) & UINT32_MASK
                self._tx_frames += 1
                return seq, ts_ms, True
            except Exception as exc:
                self._tx_errors += 1
                self._last_error = f"TX error: {exc}"
                return seq, ts_ms, False

    def send_zero_burst(self, count: int = 5, interval_s: float = 0.004) -> None:
        obs, _ = self.get_latest_observation()
        src_seq = obs.seq if obs is not None else 0
        src_ms = obs.timestamp_ms if obs is not None else 0
        zeros = (0.0,) * 6
        for i in range(max(1, count)):
            self.send_command(
                torque=zeros,
                kp=zeros,
                kd=zeros,
                q_ref=zeros,
                v_ref=zeros,
                source_seq=src_seq,
                source_timestamp_ms=src_ms,
                rl_flags=RL_FLAG_ZERO_TORQUE,
            )
            if i + 1 < count and interval_s > 0.0:
                time.sleep(interval_s)

    def stats(self) -> LinkStatsSnapshot:
        with self._rx_lock:
            return LinkStatsSnapshot(
                rx_frames=self._loss.received,
                lost_frames=self._loss.lost,
                loss_percent=self._loss.loss_percent,
                restarts=self._loss.restarts,
                duplicates=self._loss.duplicates,
                crc_errors=self._stream.crc_errors,
                length_errors=self._stream.length_errors,
                discarded_bytes=self._stream.discarded_bytes,
                raw_rx_bytes=self._raw_rx_bytes,
                tx_frames=self._tx_frames,
                tx_errors=self._tx_errors,
                connected=self._connected,
                last_error=self._last_error,
            )
