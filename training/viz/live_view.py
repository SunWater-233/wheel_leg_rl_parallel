"""Serve robot telemetry or video, PPO metrics, and playback controls on loopback."""

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from math import isfinite
from pathlib import Path
import sys
from threading import Condition, Lock, Thread
from time import monotonic, sleep


CAMERA_HOME = (111.0, -15.0, 1.0)
DRIVE_TIMEOUT = 0.5


class TrainingView:
    def __init__(self, port, *, iterations, robot, num_envs, keyboard=False, fps=None, stream=True):
        self.replay = iterations is None
        self.stream = stream
        fps = fps if fps is not None else (8 if stream and not self.replay else 60)
        self.run = {"iterations": iterations, "robot": robot, "num_envs": num_envs, "keyboard": keyboard,
                    "stream": stream, "target_fps": fps}
        self.history = deque(maxlen=300)
        self.telemetry = None
        self.terrain = None
        self.data_lock = Lock()
        self.input_lock = Lock()
        self.camera = CAMERA_HOME
        self.motion = (0, 0)
        self.motion_until = 0.0
        self.frames_changed = Condition()
        self.frame = None
        self.frame_number = 0
        self.frame_times = deque(maxlen=2 * fps + 1)
        self.error = None
        self.stopping = False
        self.stream_count = 0
        self.last_request = float("-inf")
        self.frame_interval = 1 / fps
        self.next_pose = 0.0
        self.previous_pose = None
        self.pose = None
        self.render_source = None
        page = Path(__file__).with_name("live_view.html").read_bytes()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self.send_body(200, "text/html; charset=utf-8", page)
                elif self.path == "/metrics.json":
                    self.send_body(200, "application/json; charset=utf-8", view.metrics())
                elif view.stream and self.path == "/stream.mjpg":
                    self.connection.settimeout(2.0)
                    try:
                        self.send_response(200)
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                        self.end_headers()
                        view.send_frames(self.wfile)
                    except OSError:
                        pass
                elif view.stream and self.path == "/frame.jpg":
                    with view.frames_changed:
                        view.last_request = monotonic()
                        frame, error = view.frame, view.error
                        view.frames_changed.notify_all()
                    if error is not None:
                        self.send_body(503, "text/plain; charset=utf-8", error.encode("utf-8"))
                    elif frame is None:
                        self.send_body(204, "image/jpeg", b"")
                    else:
                        self.send_body(200, "image/jpeg", frame)
                else:
                    self.send_error(404)

            def do_POST(self):
                if not view.stream or self.path != "/input":
                    self.send_error(404)
                    return
                if self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
                    self.send_error(403)
                    return
                if self.headers.get_content_type() != "application/json":
                    self.send_error(415)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 512:
                        raise ValueError("Invalid input request size")
                    view.apply_input(json.loads(self.rfile.read(size)))
                except PermissionError:
                    self.send_error(403)
                    return
                except ValueError:
                    self.send_error(400)
                    return
                self.send_body(204, "text/plain", b"")

            def send_body(self, status, content_type, body):
                try:
                    self.send_response(status)
                    self.send_header("Cache-Control", "no-store")
                    if status != 204:
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if body:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self.render_thread = Thread(target=self.render_frames, daemon=True) if stream else None

    def __enter__(self):
        self.thread.start()
        if self.render_thread is not None:
            self.render_thread.start()
        print(f"Live robot view: http://127.0.0.1:{self.server.server_port}", flush=True)
        return self

    def __exit__(self, *_):
        self.stop_rendering()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def stop_rendering(self):
        if self.render_thread is None:
            return
        with self.frames_changed:
            self.stopping = True
            self.frames_changed.notify_all()
        self.render_thread.join()

    def add_metrics(self, **values):
        with self.data_lock:
            self.history.append(self.finite_values(values))

    def set_telemetry(self, **values):
        with self.data_lock:
            self.telemetry = self.finite_values(values)

    def set_terrain(self, description):
        with self.data_lock:
            self.terrain = description

    @staticmethod
    def finite_values(values):
        return {key: value if value is not None and isfinite(value) else None for key, value in values.items()}

    def metrics(self):
        with self.data_lock:
            data = {**self.run, "history": list(self.history), "terrain": self.terrain}
            if not self.stream:
                data["telemetry"] = self.telemetry
        if self.stream:
            with self.input_lock:
                camera = self.camera
            with self.frames_changed:
                times = self.frame_times
                fresh = times and monotonic() - times[-1] < max(1.0, 2 * self.frame_interval)
                fps = (len(times) - 1) / (times[-1] - times[0]) if fresh and len(times) > 1 else None
                frame_status = {"ready": self.frame is not None, "fps": fps, "error": self.error}
            data.update(camera=camera, camera_home=CAMERA_HOME, frame=frame_status)
        return json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def apply_input(self, data):
        if not isinstance(data, dict) or not data or data.keys() - {"motion", "camera"}:
            raise ValueError("Expected robot or camera input")
        if "motion" in data:
            if not self.run["keyboard"]:
                raise PermissionError("Keyboard control requires interactive playback")
            motion = data["motion"]
            if (not isinstance(motion, list) or len(motion) != 2
                    or any(type(axis) is not int or axis not in (-1, 0, 1) for axis in motion)):
                raise ValueError("Invalid motion axes")
        if "camera" in data:
            camera = data["camera"]
            limits = ((0, 360), (-80, 60), (0.45, 3))
            if (not isinstance(camera, list) or len(camera) != 3 or any(
                type(value) not in (int, float) or not low <= value <= high
                for value, (low, high) in zip(camera, limits)
            )):
                raise ValueError("Invalid camera coordinates")
        with self.input_lock:
            if "motion" in data:
                self.motion = tuple(data["motion"])
                self.motion_until = monotonic() + DRIVE_TIMEOUT
            if "camera" in data:
                self.camera = tuple(data["camera"])

    def motion_axes(self):
        with self.input_lock:
            return self.motion if monotonic() < self.motion_until else (0, 0)

    def viewers_active(self, now):
        return self.stream_count > 0 or now - self.last_request <= 2.0

    def send_frames(self, output):
        with self.frames_changed:
            self.stream_count += 1
            self.frames_changed.notify_all()
        try:
            sent = 0
            while True:
                with self.frames_changed:
                    self.frames_changed.wait_for(lambda: self.stopping or self.error or self.frame_number != sent)
                    if self.stopping or self.error:
                        return
                    sent, frame = self.frame_number, self.frame
                header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode("ascii")
                output.write(header + frame + b"\r\n")
        finally:
            with self.frames_changed:
                self.stream_count -= 1
                self.frames_changed.notify_all()

    def capture(self, env, model_path, reset):
        now = monotonic()
        with self.frames_changed:
            if self.stopping or self.error or now < self.next_pose or not self.viewers_active(now):
                return
        qpos = env.robot.get_qpos()[0].cpu().numpy().copy()
        self.set_terrain(env.terrain_description())
        continuous = self.replay and not bool(reset)
        with self.frames_changed:
            self.previous_pose = self.pose if continuous and self.pose and now - self.pose[0] < 0.15 else None
            self.pose = (now, qpos)
            self.render_source = (model_path, env.robot, env.control_dt, getattr(env, "lanes", None))
            if not self.replay:
                self.next_pose = now + self.frame_interval
            self.frames_changed.notify_all()

    def render_frames(self):
        renderer = None
        due = monotonic()
        try:
            while True:
                with self.frames_changed:
                    self.frames_changed.wait_for(
                        lambda: self.stopping or (self.pose is not None and self.viewers_active(monotonic()))
                    )
                    if self.stopping:
                        return
                    delay = due - monotonic()
                    if delay <= 0:
                        previous = self.previous_pose
                        pose_time, qpos = self.pose
                        model_path, robot, control_dt, terrain = self.render_source
                if delay > 0:
                    sleep(delay)
                    continue
                if renderer is None:
                    from viz.render_video import RobotRenderer

                    renderer = RobotRenderer(model_path, robot, measure_closure=False, terrain=terrain)
                with self.input_lock:
                    camera = self.camera
                renderer.set_orbit(*camera)
                fraction = 1.0
                if previous is not None:
                    display_time = monotonic() - control_dt
                    fraction = max(0.0, min(1.0, (display_time - previous[0]) / (pose_time - previous[0])))
                image = renderer.render(qpos, previous=previous[1] if previous is not None else None, fraction=fraction)
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=85)
                with self.frames_changed:
                    self.frame = buffer.getvalue()
                    self.frame_number += 1
                    now = monotonic()
                    self.frame_times.append(now)
                    self.frames_changed.notify_all()
                due = max(due + self.frame_interval, now)
        except Exception as error:
            with self.frames_changed:
                self.error = f"实时画面渲染失败：{error}"
                self.frames_changed.notify_all()
            print(f"Live robot view failed: {error}", file=sys.stderr, flush=True)
        finally:
            if renderer is not None:
                renderer.close()
