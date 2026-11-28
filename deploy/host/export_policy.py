#!/usr/bin/env python3
"""Export a V0.2 WheelLegLocomotionEnv PPO checkpoint to a lightweight NumPy deploy bundle.

Outputs inside ``--output`` directory:
  - ``policy.npz``: FP32 actor MLP weights, optional obs normalizer stats, continuous
    stance lookup table ``pose_table``, and 2D V0.2 CAD forward-kinematics lookup grid
    for host-side ``leg_height_cad`` comparison.
  - ``policy_meta.json``: frozen run_config snapshot, SHA-256 digests, contract metadata,
    PD defaults, and (when PyTorch is importable) PyTorch-vs-NumPy numerical parity check.

Works both in ``training/.venv`` (using PyTorch + MuJoCo ``solve_stances`` with live
FP32 parity verification) and in pure NumPy environments (using a safe PyTorch zip
checkpoint reader + built-in Gauss-Newton V0.2 CAD closed-chain solver).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import pickle
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

HOST_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOST_DIR.parents[1]
TRAINING_DIR = REPO_ROOT / "training"

HOME_JOINTS = {
    "hip_L": -0.585,
    "hip_R": 0.585,
    "hipknee_L": -0.8201266288,
    "hipknee_R": 0.8201266397,
    "knee_L": 1.4566159,
    "knee_R": 1.8566159,
    "gas_up_L": -0.0011092105,
    "gas_up_R": 0.0011092105,
    "gas_slide_L": -0.0100408686,
    "gas_slide_R": -0.0100408686,
    "drive_l1_L": 0.2000000000,
    "drive_l1_R": -0.2000000043,
    "l1_node_L": -0.1648733712,
    "l1_node_R": 0.1648733647,
    "node_l2_L": 0.2000000000,
    "node_l2_R": -0.2000000001,
}
LEG_JOINTS = ("hip_L", "hip_R", "hipknee_L", "hipknee_R")
LEG_SIGNS = np.array((1.0, -1.0, 1.0, -1.0), dtype=np.float64)
# Raw Genesis leg_bias = (-4.25, +4.30, +4.31, -4.32) / 100.0 rad -> unified +Y direction:
LEG_BIAS_RAW_RAD = np.array((-4.25, 4.30, 4.31, -4.32), dtype=np.float64) / 100.0
LEG_BIAS_UNIFIED_RAD = LEG_BIAS_RAW_RAD * LEG_SIGNS
WHEEL_RADIUS = 0.058
WHEEL_TARGET_LIMIT = 2.9224 / WHEEL_RADIUS
STANCE_SPAWN_GAP = 0.002


# ---------------------------------------------------------------------------
# Minimal pure-Python Torch zip checkpoint loader (fallback if torch unavailable)
# ---------------------------------------------------------------------------
_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "BoolStorage": np.bool_,
}


class _StorageProxy:
    def __init__(self, array: np.ndarray):
        self.array = array


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    del requires_grad, backward_hooks, metadata
    arr = storage.array
    shape = tuple(int(s) for s in size)
    if not shape:
        return np.array(arr[int(storage_offset)], copy=True)
    itemsize = arr.dtype.itemsize
    byte_strides = tuple(int(st) * itemsize for st in stride)
    base = arr[int(storage_offset):]
    view = np.lib.stride_tricks.as_strided(base, shape=shape, strides=byte_strides)
    return np.array(view, copy=True)


def _rebuild_tensor(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2(storage, storage_offset, size, stride, False, None)


class _DummyOrderedDict(dict):
    pass


class _PureTorchUnpickler(pickle.Unpickler):
    def __init__(self, bio: io.BytesIO, zf: zipfile.ZipFile, archive_prefix: str):
        super().__init__(bio)
        self._zf = zf
        self._prefix = archive_prefix
        self._storages: dict[str, _StorageProxy] = {}

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            storage_type, key, _location, numel = pid[1], str(pid[2]), pid[3], int(pid[4])
            if key not in self._storages:
                dtype = _STORAGE_DTYPES.get(storage_type, np.float32)
                raw = self._zf.read(f"{self._prefix}/data/{key}")
                arr = np.frombuffer(raw, dtype=dtype, count=numel)
                self._storages[key] = _StorageProxy(arr)
            return self._storages[key]
        return None

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor_v2 if name == "_rebuild_tensor_v2" else _rebuild_tensor
        if module == "torch" and name in _STORAGE_DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return _DummyOrderedDict
        if module == "builtins":
            return getattr(__builtins__, name, None) or __builtins__[name]
        # Ignore optimizer/device/custom metadata classes cleanly
        return lambda *args, **kwargs: None


def load_checkpoint_arrays(checkpoint_path: Path) -> tuple[dict[str, np.ndarray], dict, str]:
    """Load model_state_dict arrays and infos dict from a RSL-RL .pt checkpoint."""
    try:
        import torch  # type: ignore

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = {
            k: v.detach().cpu().numpy()
            for k, v in ckpt["model_state_dict"].items()
            if hasattr(v, "detach")
        }
        infos = ckpt.get("infos") or {}
        return state, infos, f"torch {torch.__version__}"
    except Exception:
        with zipfile.ZipFile(checkpoint_path, "r") as zf:
            pkl_names = [n for n in zf.namelist() if n.endswith("/data.pkl")]
            if not pkl_names:
                raise ValueError(f"Unsupported checkpoint archive (missing data.pkl): {checkpoint_path}")
            prefix = pkl_names[0].rsplit("/", 1)[0]
            data = zf.read(pkl_names[0])
            unpickler = _PureTorchUnpickler(io.BytesIO(data), zf, prefix)
            ckpt = unpickler.load()
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
        state = {
            k: np.asarray(v)
            for k, v in ckpt["model_state_dict"].items()
            if isinstance(v, np.ndarray)
        }
        infos = ckpt.get("infos") if isinstance(ckpt.get("infos"), dict) else {}
        return state, infos, "pure-numpy-zip"


# ---------------------------------------------------------------------------
# Pure-NumPy V0.2 CAD Closed-Chain Kinematics & Stance Solver
# ---------------------------------------------------------------------------
def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _qrot(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _axis_angle_quat(axis, angle: float):
    ax, ay, az = axis
    half = 0.5 * angle
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


class V02CadKinematics:
    """Fast 3D kinematic tree + closed-chain loop solver parsed directly from wheel_leg_ground.xml."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        root = ET.parse(self.model_path).getroot()
        base = root.find('.//body[@name="base_link"]')
        if base is None:
            raise ValueError(f"base_link not found in {model_path}")

        self.bodies: list[dict] = []
        self.body_idx: dict[str, int] = {}
        self.joints: dict[str, dict] = {}
        self.sites: dict[str, tuple[int, tuple[float, float, float]]] = {}

        def parse_body(elem: ET.Element, parent_idx: int) -> None:
            idx = len(self.bodies)
            name = elem.attrib["name"]
            pos = tuple(float(v) for v in elem.attrib.get("pos", "0 0 0").split())
            q = [float(v) for v in elem.attrib.get("quat", "1 0 0 0").split()]
            qn = math.sqrt(sum(v * v for v in q))
            quat = tuple(v / qn for v in q)
            body_joints = []
            for j in elem.findall("joint"):
                jn = j.attrib["name"]
                jtype = j.attrib.get("type", "hinge")
                if jtype == "free":
                    continue
                jpos = tuple(float(v) for v in j.attrib.get("pos", "0 0 0").split())
                ja = [float(v) for v in j.attrib.get("axis", "0 0 1").split()]
                jan = math.sqrt(sum(v * v for v in ja))
                jaxis = tuple(v / jan for v in ja)
                ref = float(j.attrib.get("ref", "0.0"))
                rng = j.attrib.get("range")
                limits = tuple(float(v) for v in rng.split()) if rng and j.attrib.get("limited") == "true" else (-np.inf, np.inf)
                jinfo = {
                    "name": jn,
                    "type": jtype,
                    "pos": jpos,
                    "axis": jaxis,
                    "ref": ref,
                    "limits": limits,
                    "body_idx": idx,
                }
                body_joints.append(jinfo)
                self.joints[jn] = jinfo
            self.bodies.append({
                "name": name,
                "parent": parent_idx,
                "pos": pos,
                "quat": quat,
                "joints": body_joints,
            })
            self.body_idx[name] = idx
            for s in elem.findall("site"):
                spos = tuple(float(v) for v in s.attrib.get("pos", "0 0 0").split())
                self.sites[s.attrib["name"]] = (idx, spos)
            for child in elem.findall("body"):
                parse_body(child, idx)

        parse_body(base, -1)

        # Parse planar connect constraints (minus sites suffice since linkage is planar in X-Z)
        self.connect_pairs: dict[str, list[tuple[str, str]]] = {"L": [], "R": []}
        for conn in root.findall(".//equality/connect"):
            cname = conn.attrib["name"]
            if cname.endswith("_minus") and not cname.startswith("gas_"):
                side = "L" if "_L_" in cname else "R"
                self.connect_pairs[side].append((conn.attrib["site1"], conn.attrib["site2"]))

        self.side_passive = {
            side: (f"knee_{side}", f"drive_l1_{side}", f"l1_node_{side}", f"node_l2_{side}")
            for side in ("L", "R")
        }

        # Precompute minimal ordered body subset per side (only ~8 bodies instead of all ~35)
        self.side_body_order: dict[str, list[int]] = {}
        for side in ("L", "R"):
            needed: set[int] = set()
            targets = [
                self.joints[f"hip_{side}"]["body_idx"],
                self.joints[f"wheel_{side}"]["body_idx"],
            ]
            for s1, s2 in self.connect_pairs[side]:
                targets.append(self.sites[s1][0])
                targets.append(self.sites[s2][0])
            for bidx in targets:
                cur = bidx
                while cur >= 0 and cur not in needed:
                    needed.add(cur)
                    cur = self.bodies[cur]["parent"]
            self.side_body_order[side] = sorted(needed)

    def fk_side(self, side: str, q_dict: dict[str, float]) -> tuple[dict[int, tuple], dict[int, tuple]]:
        """Forward kinematics for ONLY the ~8 bodies of one leg in base_link frame."""
        bpos: dict[int, tuple] = {}
        bquat: dict[int, tuple] = {}
        for i in self.side_body_order[side]:
            body = self.bodies[i]
            p_idx = body["parent"]
            if p_idx < 0:
                pos = (0.0, 0.0, 0.0)
                quat = (1.0, 0.0, 0.0, 0.0)
            else:
                pp, pq = bpos[p_idx], bquat[p_idx]
                rp = _qrot(pq, body["pos"])
                pos = (pp[0] + rp[0], pp[1] + rp[1], pp[2] + rp[2])
                quat = _qmul(pq, body["quat"])
            for j in body["joints"]:
                qval = float(q_dict.get(j["name"], HOME_JOINTS.get(j["name"], 0.0)))
                dq = qval - j["ref"]
                if j["type"] == "hinge":
                    rj = _qrot(quat, j["pos"])
                    aw = (pos[0] + rj[0], pos[1] + rj[1], pos[2] + rj[2])
                    axis_w = _qrot(quat, j["axis"])
                    quat = _qmul(_axis_angle_quat(axis_w, dq), quat)
                    rj2 = _qrot(quat, j["pos"])
                    pos = (aw[0] - rj2[0], aw[1] - rj2[1], aw[2] - rj2[2])
                elif j["type"] == "slide":
                    ax = j["axis"]
                    rp = _qrot(quat, (ax[0] * dq, ax[1] * dq, ax[2] * dq))
                    pos = (pos[0] + rp[0], pos[1] + rp[1], pos[2] + rp[2])
            bpos[i] = pos
            bquat[i] = quat
        return bpos, bquat

    def site_pos(self, bpos, bquat, site_name: str) -> tuple[float, float, float]:
        idx, local = self.sites[site_name]
        bp, bq = bpos[idx], bquat[idx]
        rl = _qrot(bq, local)
        return (bp[0] + rl[0], bp[1] + rl[1], bp[2] + rl[2])

    def joint_anchor(self, bpos, bquat, joint_name: str) -> tuple[float, float, float]:
        j = self.joints[joint_name]
        idx = j["body_idx"]
        bp, bq = bpos[idx], bquat[idx]
        rl = _qrot(bq, j["pos"])
        return (bp[0] + rl[0], bp[1] + rl[1], bp[2] + rl[2])

    def solve_side_fk(
        self,
        side: str,
        hip_val: float,
        hipknee_val: float,
        init_passive: np.ndarray | None = None,
        *,
        max_iter: int = 20,
        tol: float = 1e-7,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Given driven (hip, hipknee), solve the 4 passive angles and return (hip->wheel vector xz, passive, residual)."""
        pnames = self.side_passive[side]
        pairs = self.connect_pairs[side]
        x = (
            np.array([HOME_JOINTS[n] for n in pnames], dtype=np.float64)
            if init_passive is None
            else np.array(init_passive, dtype=np.float64)
        )
        lows = np.array([self.joints[n]["limits"][0] for n in pnames], dtype=np.float64)
        highs = np.array([self.joints[n]["limits"][1] for n in pnames], dtype=np.float64)
        hip_name = f"hip_{side}"
        hk_name = f"hipknee_{side}"
        qd = {hip_name: hip_val, hk_name: hipknee_val}

        def eval_res(pv: np.ndarray) -> np.ndarray:
            for nm, v in zip(pnames, pv):
                qd[nm] = float(v)
            bp, bq = self.fk_side(side, qd)
            out = []
            for s1, s2 in pairs:
                p1 = self.site_pos(bp, bq, s1)
                p2 = self.site_pos(bp, bq, s2)
                out.extend((p1[0] - p2[0], p1[2] - p2[2]))
            return np.array(out, dtype=np.float64)

        eps = 1e-6
        lam = 1e-8
        r = eval_res(x)
        for _ in range(max_iter):
            err = float(np.max(np.abs(r)))
            if err < tol:
                break
            J = np.empty((len(r), len(x)), dtype=np.float64)
            for k in range(len(x)):
                xp = x.copy()
                xp[k] += eps
                J[:, k] = (eval_res(xp) - r) / eps
            A = J.T @ J + lam * np.eye(len(x))
            g = J.T @ r
            try:
                dx = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            cand = np.clip(x + dx, lows, highs)
            rc = eval_res(cand)
            if float(np.dot(rc, rc)) < float(np.dot(r, r)):
                x, r = cand, rc
                lam = max(lam * 0.3, 1e-10)
            else:
                lam *= 5.0

        for nm, v in zip(pnames, x):
            qd[nm] = float(v)
        bp, bq = self.fk_side(side, qd)
        hip_p = self.joint_anchor(bp, bq, hip_name)
        wheel_p = self.joint_anchor(bp, bq, f"wheel_{side}")
        leg_vec = np.array([hip_p[0] - wheel_p[0], hip_p[2] - wheel_p[2]], dtype=np.float64)
        return leg_vec, x, float(np.max(np.abs(r)))

    def solve_stances_pure_numpy(
        self,
        height_limits: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve nominal leg angles [hip_L, hip_R, hipknee_L, hipknee_R] across continuous heights."""
        low_h, high_h = float(height_limits[0]), float(height_limits[1])
        n_pts = max(2, math.ceil((high_h - low_h) / 0.005) + 1)
        heights = np.linspace(low_h, high_h, n_pts, dtype=np.float64)

        # Solve home reference horizontal offset
        home_vec_L, home_pass_L, err_L = self.solve_side_fk("L", HOME_JOINTS["hip_L"], HOME_JOINTS["hipknee_L"])
        home_vec_R, home_pass_R, err_R = self.solve_side_fk("R", HOME_JOINTS["hip_R"], HOME_JOINTS["hipknee_R"])
        if max(err_L, err_R) > 1e-4:
            raise RuntimeError(f"CAD home closure error too large: L={err_L:.3e}, R={err_R:.3e}")

        nominal_legs = np.empty((n_pts, 4), dtype=np.float64)
        for s_idx, (side, h_col, hk_col, home_vec, home_pass) in enumerate((
            ("L", 0, 2, home_vec_L, home_pass_L),
            ("R", 1, 3, home_vec_R, home_pass_R),
        )):
            del s_idx
            target_x = float(home_vec[0])
            order = np.argsort(np.abs(heights - float(home_vec[1])))
            q_cur = np.array([HOME_JOINTS[f"hip_{side}"], HOME_JOINTS[f"hipknee_{side}"]], dtype=np.float64)
            p_cur = home_pass.copy()
            for idx in order:
                h_tgt = float(heights[idx])
                lam = 1e-5
                for _ in range(40):
                    vec, p_cur, cerr = self.solve_side_fk(side, q_cur[0], q_cur[1], p_cur)
                    res = np.array([
                        vec[1] - h_tgt,
                        0.1 * (vec[0] - target_x),
                        0.001 * (q_cur[0] - HOME_JOINTS[f"hip_{side}"]),
                        0.001 * (q_cur[1] - HOME_JOINTS[f"hipknee_{side}"]),
                    ], dtype=np.float64)
                    if abs(vec[1] - h_tgt) < 1e-6 and abs(vec[0] - target_x) < 1e-4 and cerr < 1e-5:
                        break
                    eps = 1e-5
                    J = np.empty((4, 2), dtype=np.float64)
                    for k in range(2):
                        qp = q_cur.copy()
                        qp[k] += eps
                        vp, _, _ = self.solve_side_fk(side, qp[0], qp[1], p_cur)
                        rp = np.array([
                            vp[1] - h_tgt,
                            0.1 * (vp[0] - target_x),
                            0.001 * (qp[0] - HOME_JOINTS[f"hip_{side}"]),
                            0.001 * (qp[1] - HOME_JOINTS[f"hipknee_{side}"]),
                        ], dtype=np.float64)
                        J[:, k] = (rp - res) / eps
                    dq = -np.linalg.solve(J.T @ J + lam * np.eye(2), J.T @ res)
                    dq = np.clip(dq, -0.25, 0.25)
                    q_cand = q_cur + dq
                    vc, pc, _ = self.solve_side_fk(side, q_cand[0], q_cand[1], p_cur)
                    rc = np.array([
                        vc[1] - h_tgt,
                        0.1 * (vc[0] - target_x),
                        0.001 * (q_cand[0] - HOME_JOINTS[f"hip_{side}"]),
                        0.001 * (q_cand[1] - HOME_JOINTS[f"hipknee_{side}"]),
                    ], dtype=np.float64)
                    if float(np.dot(rc, rc)) < float(np.dot(res, res)):
                        q_cur, p_cur = q_cand, pc
                        lam = max(lam * 0.5, 1e-8)
                    else:
                        lam *= 5.0
                vec, p_cur, cerr = self.solve_side_fk(side, q_cur[0], q_cur[1], p_cur)
                if cerr > 5e-5 or abs(vec[1] - h_tgt) > 5e-4:
                    raise RuntimeError(
                        f"Pure-NumPy CAD stance failed at h={h_tgt:.3f}m ({side}): "
                        f"cerr={cerr:.2e}, dh={abs(vec[1] - h_tgt):.2e}"
                    )
                nominal_legs[idx, h_col] = q_cur[0]
                nominal_legs[idx, hk_col] = q_cur[1]

        return heights, nominal_legs

    def build_cad_fk_grid(
        self,
        hip_range: tuple[float, float] = (-1.65, 0.35),
        hipknee_range: tuple[float, float] = (-1.95, 0.15),
        grid_n: int = 25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Precompute 2D lookup grid of (hip - wheel) [dx, dz] in base_link frame for unified +Y angles.

        Evaluated on the left leg (whose CAD axes are +Y, i.e. unified +Y coordinates).
        Right leg uses the exact same sagittal geometry after sign unification.
        """
        hip_grid = np.linspace(hip_range[0], hip_range[1], grid_n, dtype=np.float64)
        hk_grid = np.linspace(hipknee_range[0], hipknee_range[1], grid_n, dtype=np.float64)
        vec_x = np.full((grid_n, grid_n), np.nan, dtype=np.float64)
        vec_z = np.full((grid_n, grid_n), np.nan, dtype=np.float64)

        i0 = int(np.argmin(np.abs(hip_grid - HOME_JOINTS["hip_L"])))
        j0 = int(np.argmin(np.abs(hk_grid - HOME_JOINTS["hipknee_L"])))
        _, p0, _ = self.solve_side_fk("L", float(hip_grid[i0]), float(hk_grid[j0]))

        # BFS outward from home so continuation stays on the physical assembly branch
        from collections import deque

        visited = np.zeros((grid_n, grid_n), dtype=bool)
        passive_map: dict[tuple[int, int], np.ndarray] = {(i0, j0): p0}
        q = deque([(i0, j0)])
        visited[i0, j0] = True
        while q:
            i, j = q.popleft()
            p_init = passive_map[(i, j)]
            vec, p_sol, cerr = self.solve_side_fk("L", float(hip_grid[i]), float(hk_grid[j]), p_init, max_iter=12, tol=1e-6)
            if cerr < 1e-3:
                vec_x[i, j] = vec[0]
                vec_z[i, j] = vec[1]
                passive_map[(i, j)] = p_sol
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_n and 0 <= nj < grid_n and not visited[ni, nj]:
                        visited[ni, nj] = True
                        passive_map[(ni, nj)] = p_sol
                        q.append((ni, nj))
        return hip_grid, hk_grid, vec_x, vec_z


def solve_nominal_legs(model_path: Path, height_limits: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute nominal leg angles [hip_L, hip_R, hipknee_L, hipknee_R] across height_limits."""
    try:
        sys.path.insert(0, str(TRAINING_DIR))
        from robots.wheel_leg_stance import solve_stances  # type: ignore

        poses = solve_stances(model_path, height_limits, WHEEL_RADIUS)
        leg_cols = [list(HOME_JOINTS).index(name) + 1 for name in LEG_JOINTS]
        nominal_legs = np.asarray(poses[:, leg_cols], dtype=np.float64)
        heights = np.linspace(float(height_limits[0]), float(height_limits[1]), len(poses), dtype=np.float64)
        return heights, nominal_legs, "mujoco"
    except Exception:
        cad = V02CadKinematics(model_path)
        heights, nominal_legs = cad.solve_stances_pure_numpy(height_limits)
        return heights, nominal_legs, "pure-numpy-cad"


def extract_actor_layers(state_dict: dict[str, np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pattern = re.compile(r"^actor\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for k in state_dict if (m := pattern.match(k)))
    if not indices:
        raise ValueError("No actor.<idx>.weight tensors found in checkpoint state_dict")
    weights = [np.asarray(state_dict[f"actor.{i}.weight"], dtype=np.float32) for i in indices]
    biases = [np.asarray(state_dict[f"actor.{i}.bias"], dtype=np.float32) for i in indices]
    return weights, biases


def extract_obs_normalizer(state_dict: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    mean = None
    std = None
    eps = 1e-2
    for key in ("actor_obs_normalizer._mean", "obs_norm._mean", "actor_obs_normalizer.mean"):
        if key in state_dict:
            mean = np.asarray(state_dict[key], dtype=np.float32).reshape(-1)
            break
    for key in ("actor_obs_normalizer._std", "obs_norm._std", "actor_obs_normalizer.std"):
        if key in state_dict:
            std = np.asarray(state_dict[key], dtype=np.float32).reshape(-1)
            break
    if mean is not None and std is None:
        for key in ("actor_obs_normalizer._var", "obs_norm._var", "actor_obs_normalizer.var"):
            if key in state_dict:
                std = np.sqrt(np.asarray(state_dict[key], dtype=np.float32).reshape(-1) + 1e-8)
                break
    return mean, std, eps


def _numpy_forward(
    x: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation: str,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    norm_eps: float = 1e-2,
) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    if norm_mean is not None and norm_std is not None:
        h = (h - norm_mean) / (norm_std + np.float32(norm_eps))
    act = activation.lower()
    for idx, (w, b) in enumerate(zip(weights, biases)):
        h = h @ w.T + b
        if idx < len(weights) - 1:
            if act == "elu":
                h = np.where(h > 0.0, h, np.expm1(np.clip(h, -80.0, 0.0))).astype(np.float32)
            elif act == "relu":
                h = np.maximum(h, 0.0).astype(np.float32)
            elif act == "tanh":
                h = np.tanh(h).astype(np.float32)
            elif act in ("silu", "swish"):
                h = (h / (1.0 + np.exp(-np.clip(h, -80.0, 80.0)))).astype(np.float32)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
    return h


def verify_torch_parity(
    checkpoint_path: Path,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation: str,
    norm_mean: np.ndarray | None,
    norm_std: np.ndarray | None,
    norm_eps: float,
) -> float | None:
    """If PyTorch is available, verify NumPy forward matches Torch MLP to < 1e-5."""
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception:
        return None

    act_map = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}
    act_cls = act_map.get(activation.lower())
    if act_cls is None:
        return None
    layers = []
    for i, (w, b) in enumerate(zip(weights, biases)):
        lin = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
        layers.append(lin)
        if i < len(weights) - 1:
            layers.append(act_cls())
    net = nn.Sequential(*layers).eval()

    rng = np.random.default_rng(42)
    x = rng.standard_normal((32, weights[0].shape[1])).astype(np.float32)
    y_np = _numpy_forward(x, weights, biases, activation, norm_mean, norm_std, norm_eps)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        if norm_mean is not None and norm_std is not None:
            xt = (xt - torch.from_numpy(norm_mean)) / (torch.from_numpy(norm_std) + norm_eps)
        y_th = net(xt).numpy()
    max_err = float(np.max(np.abs(y_np - y_th)))
    if max_err > 1e-4:
        raise RuntimeError(f"Torch-vs-NumPy parity check failed: max_err={max_err:.3e}")
    return max_err


def export_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    run_config_path: Path | None = None,
    model_xml_path: Path | None = None,
    skip_cad_grid: bool = False,
) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg_path = run_config_path or checkpoint_path.with_name("run_config.json")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"run_config.json not found next to checkpoint: {cfg_path}")
    run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    task = run_cfg["task"]
    loco = task["locomotion"]
    ppo_policy = run_cfg["ppo"]["policy"]
    run_info = run_cfg.get("run", {})

    if loco.get("actor_true_forward_speed", False):
        raise ValueError("Cannot deploy a policy trained with privileged actor_true_forward_speed=True")

    model_rel = run_info.get("model", "assets/wheel_leg_xml_V0.2/wheel_leg_ground.xml")
    xml_path = (model_xml_path or (TRAINING_DIR / model_rel)).resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"V0.2 MJCF model not found: {xml_path}")

    state_dict, infos, ckpt_loader = load_checkpoint_arrays(checkpoint_path)
    weights, biases = extract_actor_layers(state_dict)
    norm_mean, norm_std, norm_eps = extract_obs_normalizer(state_dict)
    activation = str(ppo_policy.get("activation", "elu"))

    history_steps = int(loco.get("actor_history_steps", 1))
    single_obs_dim = 28
    expected_in_dim = single_obs_dim * history_steps
    actual_in_dim = int(weights[0].shape[1])
    actual_out_dim = int(weights[-1].shape[0])
    if actual_in_dim != expected_in_dim:
        raise ValueError(
            f"Actor input dim mismatch: weights expect {actual_in_dim}, "
            f"but 28 * actor_history_steps({history_steps}) = {expected_in_dim}"
        )
    if actual_out_dim != 6:
        raise ValueError(f"Actor output dim mismatch: expected 6, got {actual_out_dim}")

    height_range = (float(loco["height_range"][0]), float(loco["height_range"][1]))
    pose_heights, nominal_legs_raw, stance_solver = solve_nominal_legs(xml_path, height_range)
    # Convert nominal leg angles into unified +Y coordinates (matching firmware reversed=true)
    nominal_legs_unified = nominal_legs_raw * LEG_SIGNS[None, :]

    cad_grid = None
    if not skip_cad_grid:
        cad = V02CadKinematics(xml_path)
        cad_grid = cad.build_cad_fk_grid()

    parity_max_err = verify_torch_parity(
        checkpoint_path, weights, biases, activation, norm_mean, norm_std, norm_eps
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_payload: dict[str, np.ndarray] = {
        "num_layers": np.array(len(weights), dtype=np.int32),
        "pose_heights": pose_heights.astype(np.float32),
        "nominal_legs_raw": nominal_legs_raw.astype(np.float32),
        "nominal_legs_unified": nominal_legs_unified.astype(np.float32),
        "leg_signs": LEG_SIGNS.astype(np.float32),
        "leg_bias_raw": LEG_BIAS_RAW_RAD.astype(np.float32),
        "leg_bias_unified": LEG_BIAS_UNIFIED_RAD.astype(np.float32),
    }
    for i, (w, b) in enumerate(zip(weights, biases)):
        npz_payload[f"w_{i}"] = w.astype(np.float32)
        npz_payload[f"b_{i}"] = b.astype(np.float32)
    if norm_mean is not None and norm_std is not None:
        npz_payload["obs_norm_mean"] = norm_mean.astype(np.float32)
        npz_payload["obs_norm_std"] = norm_std.astype(np.float32)
        npz_payload["obs_norm_eps"] = np.array(norm_eps, dtype=np.float32)
    if cad_grid is not None:
        hip_grid, hk_grid, vec_x, vec_z = cad_grid
        npz_payload["cad_hip_grid"] = hip_grid.astype(np.float32)
        npz_payload["cad_hk_grid"] = hk_grid.astype(np.float32)
        npz_payload["cad_vec_x"] = vec_x.astype(np.float32)
        npz_payload["cad_vec_z"] = vec_z.astype(np.float32)

    npz_path = output_dir / "policy.npz"
    np.savez(npz_path, **npz_payload)

    meta = {
        "bundle_version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path.read_bytes()).hexdigest(),
        "model_xml_path": str(xml_path),
        "model_xml_sha256": sha256(xml_path.read_bytes()).hexdigest(),
        "checkpoint_loader": ckpt_loader,
        "stance_solver": stance_solver,
        "torch_parity_max_abs_err": parity_max_err,
        "checkpoint_infos": {k: str(v) for k, v in infos.items()},
        "contract": {
            "robot": run_info.get("robot", "v0.2"),
            "task": run_info.get("task", "locomotion"),
            "policy_order": ["hip_L", "hip_R", "hipknee_L", "hipknee_R", "wheel_L", "wheel_R"],
            "single_obs_dim": single_obs_dim,
            "actor_history_steps": history_steps,
            "actor_input_dim": expected_in_dim,
            "action_dim": 6,
            "activation": activation,
            "hidden_dims": [int(w.shape[0]) for w in weights[:-1]],
            "has_obs_normalizer": norm_mean is not None and norm_std is not None,
            "has_cad_fk_grid": cad_grid is not None,
            "physics_dt": float(run_info.get("physics_dt", 0.002)),
            "decimation": int(run_info.get("decimation", 10)),
            "control_dt": float(run_info.get("control_dt", 0.02)),
            "forward_command": [float(v) for v in task.get("forward_command", [-2.5, 2.5])],
            "yaw_command": [float(v) for v in task.get("yaw_command", [-10.0, 10.0])],
            "height_range": list(height_range),
            "height_nominal": float(loco.get("height_nominal", 0.20)),
            "height_rate": float(loco.get("height_rate", 0.35)),
            "height_acceleration": float(loco.get("height_acceleration", 1.2)),
            "wheel_radius": WHEEL_RADIUS,
            "half_track": 0.21865,
            "leg_action_scale": 0.20,
            "leg_offset_limit": 0.35,
            "wheel_action_scale": 10.0,
            "wheel_target_limit": WHEEL_TARGET_LIMIT,
            "leg_torque_limit": float(loco.get("leg_torque_limit", 40.0)),
            "wheel_torque_limit": float(loco.get("wheel_torque_limit", 5.0)),
            "wheel_target_delay_steps": int(loco.get("wheel_target_delay_steps", 5)),
            "default_pd_gains": {
                "leg_kp": [100.0, 100.0, 100.0, 100.0],
                "leg_kd": [4.0, 4.0, 4.0, 4.0],
                "wheel_kd": [0.6, 0.6],
            },
        },
        "run_config_snapshot": run_cfg,
    }
    meta_path = output_dir / "policy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model_XXXX.pt")
    parser.add_argument("--output", type=Path, help="Output bundle directory (default: deploy/host/bundles/<run>_<ckpt>)")
    parser.add_argument("--run-config", type=Path, help="Optional explicit run_config.json path")
    parser.add_argument("--model-xml", type=Path, help="Optional explicit wheel_leg_ground.xml path")
    parser.add_argument("--skip-cad-grid", action="store_true", help="Skip 2D CAD FK lookup grid generation")
    args = parser.parse_args()

    out = args.output
    if out is None:
        ckpt = args.checkpoint.resolve()
        out = HOST_DIR / "bundles" / f"{ckpt.parent.name}_{ckpt.stem}"

    bundle_dir = export_bundle(
        args.checkpoint,
        out,
        run_config_path=args.run_config,
        model_xml_path=args.model_xml,
        skip_cad_grid=args.skip_cad_grid,
    )
    meta = json.loads((bundle_dir / "policy_meta.json").read_text(encoding="utf-8"))
    c = meta["contract"]
    print(
        f"Exported deploy bundle -> {bundle_dir}\n"
        f"  loader={meta['checkpoint_loader']}, stance={meta['stance_solver']}, "
        f"torch_parity={meta['torch_parity_max_abs_err']}\n"
        f"  obs={c['single_obs_dim']}x{c['actor_history_steps']}={c['actor_input_dim']}D -> "
        f"MLP{c['hidden_dims']}({c['activation']}) -> {c['action_dim']}D\n"
        f"  height={c['height_range']} m, limits=({c['leg_torque_limit']} / {c['wheel_torque_limit']} N.m), "
        f"cad_grid={c['has_cad_fk_grid']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
