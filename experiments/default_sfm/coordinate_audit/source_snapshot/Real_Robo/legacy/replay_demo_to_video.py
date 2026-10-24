#!/usr/bin/env python3
"""Offline, hardware-free replay of a saved demo pickle to an MP4.

Loads a demo file produced by `demo.py` / `demo_pika.py`, pulls the recorded
Cartesian `tcp_pose` for each arm out of `obs["state"]`, solves IK against
the same MuJoCo model the real pipeline uses (`DualRb3Mujoco`), and draws a
3D skeleton (body-tree line segments + TCP markers) of the resulting pose
per timestep with matplotlib, saving the frames to an MP4. No ROS node, no
real robot/gripper connection, no display, and no OpenGL/EGL context are
required -- this deliberately avoids `mujoco.Renderer` (which needs a
working GL backend) so it runs headless with nothing beyond what a normal
ML conda env already has (mujoco, mink, scipy, matplotlib), piping frames
into the system `ffmpeg` binary to encode H.264 (OpenCV's `mp4v` fourcc
produces a codec most browsers/VSCode's built-in video preview can't play).

Known limitations:
- This is a kinematic reconstruction, not a replay of the real robot's
  actual joint angles: only the recorded TCP (end-effector) pose is real
  data, and joint angles are re-derived via IK, so the joint-space path may
  differ from what the real robot actually did even though the TCP path
  matches. (`current_qpos` is not currently saved in demo pickles --
  `DualRb3Env.step()` drops `infos` -- so exact joint replay isn't possible
  from existing data.)
- It's a 3D line-skeleton, not a meshed/textured render.
- Only arm joints are driven from IK; gripper finger joints are left at the
  model's default qpos (opening/closing is not animated).
- `obs["state"]` is assumed to carry a single most-recent frame per side; if
  it is 2D (from ChunkingWrapper with obs_horizon > 1), the LAST row is used.
- `--proprio-keys` must match `train.proprio_keys` from the config used at
  collection time, so the flat `state` vector can be sliced back apart.

Run inside an environment with `mujoco`, `mink`, `scipy`, and `matplotlib`
installed (e.g. the `umi` conda env), not plain system python3. Do NOT
set MUJOCO_GL -- leaving it unset is what lets `import mujoco` succeed
without touching any GL backend. Requires `ffmpeg` on PATH.
"""
import argparse
import pickle
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from external.dual_rb3_mujoco.dual_rb3_mujoco.dual_rb3_mujoco import (  # noqa: E402
    DualRb3Mujoco,
)
from ..resource_paths import mujoco_model_path

DEFAULT_PROPRIO_KEYS = [
    "left/gripper_pose", "left/tcp_force", "left/tcp_pose",
    "left/tcp_torque", "left/tcp_vel",
    "right/gripper_pose", "right/tcp_force", "right/tcp_pose",
    "right/tcp_torque", "right/tcp_vel",
]

# Field widths as exposed by Rb3Env.observation_space, AFTER
# SidePrefixedQuat2EulerWrapper shrinks tcp_pose from 7 to 6 values.
PROPRIO_DIMS = {
    "gripper_pose": 1,
    "tcp_force": 3,
    "tcp_pose": 6,
    "tcp_torque": 3,
    "tcp_vel": 6,
}


def build_state_index(proprio_keys):
    """gym.spaces.Dict flattens its sub-spaces sorted by key; reproduce that
    ordering so the flat `state` vector can be sliced back out by name."""
    offsets = {}
    cursor = 0
    for key in sorted(proprio_keys):
        _, field = key.split("/", 1)
        dim = PROPRIO_DIMS[field]
        offsets[key] = slice(cursor, cursor + dim)
        cursor += dim
    return offsets


def euler_pose_to_scalar_first_quat_pose(pose6):
    """[x,y,z, rx,ry,rz] -> [x,y,z, w,qx,qy,qz], matching pose_to_se3()."""
    xyz = pose6[:3]
    quat_xyzw = R.from_euler("xyz", pose6[3:]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return np.concatenate([xyz, quat_wxyz])


class SkeletonRenderer:
    """Draws the model's dual_rb3/* body tree + TCP sites as a 3D line
    skeleton with matplotlib (Agg backend -- pure software, no GL)."""

    def __init__(self, model, width, height, lookat, elevation, azimuth, radius):
        self.model = model
        dpi = 100
        self.fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.lookat = np.asarray(lookat)
        self.elevation = elevation
        self.azimuth = azimuth
        self.radius = radius

        # (parent_body_id, child_body_id) edges restricted to the robot's
        # own bodies -- this generically reflects whatever gripper/tip is
        # actually attached in the loaded MJCF, no hardcoded link names.
        self.edges = []
        for i in range(model.nbody):
            name = model.body(i).name
            if not name.startswith("dual_rb3/"):
                continue
            parent = int(model.body(i).parentid[0])
            self.edges.append((parent, i, "left" if "/left/" in name else "right"))

        self.tcp_site_ids = {}
        for side in ("left", "right"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"dual_rb3/{side}/tcp")
            if sid >= 0:
                self.tcp_site_ids[side] = sid

    def render(self, data, title=""):
        ax = self.ax
        ax.cla()
        for parent, child, side in self.edges:
            p1, p2 = data.xpos[parent], data.xpos[child]
            color = "tab:blue" if side == "left" else "tab:orange"
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=2)

        for side, sid in self.tcp_site_ids.items():
            p = data.site_xpos[sid]
            color = "red" if side == "left" else "green"
            ax.scatter([p[0]], [p[1]], [p[2]], color=color, s=50, label=f"{side} tcp")

        lx, ly, lz = self.lookat
        r = self.radius
        ax.set_xlim(lx - r, lx + r)
        ax.set_ylim(ly - r, ly + r)
        ax.set_zlim(max(lz - r, 0), lz + r)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=self.elevation, azim=self.azimuth)
        ax.legend(loc="upper left", fontsize=8)
        if title:
            ax.set_title(title, fontsize=9)

        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    def frame_size(self):
        return self.fig.canvas.get_width_height()


class FfmpegVideoWriter:
    """Pipes raw RGB24 frames into ffmpeg's libx264 encoder so the output
    plays in browsers / VSCode's built-in preview (unlike OpenCV's default
    'mp4v' fourcc, which most of those can't decode)."""

    def __init__(self, output_path, width, height, fps):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb_frame):
        self.proc.stdin.write(rgb_frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demo_pkl", help="*_demos_*.pkl saved by demo.py/demo_pika.py")
    parser.add_argument(
        "mjcf",
        nargs="?",
        default=str(mujoco_model_path("scene_left_pika_right_dh.xml")),
        help="MJCF scene matching the recording (defaults to the packaged left-PIKA/right-DH scene)",
    )
    parser.add_argument("-o", "--output", default="replay.mp4")
    parser.add_argument("--fps", type=float, default=10.0, help="should match train.hz used to collect")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--proprio-keys", nargs="+", default=DEFAULT_PROPRIO_KEYS)
    parser.add_argument("--lookat", nargs=3, type=float, default=[0.3, 0.0, 0.6])
    parser.add_argument("--view-radius", type=float, default=0.8)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    args = parser.parse_args()

    with open(args.demo_pkl, "rb") as f:
        transitions = pickle.load(f)
    transitions = transitions[args.start:]
    if args.max_steps:
        transitions = transitions[: args.max_steps]
    print(f"loaded {len(transitions)} transitions from {args.demo_pkl}")

    state_index = build_state_index(args.proprio_keys)

    robot = DualRb3Mujoco(args.mjcf)
    mujoco.mj_resetData(robot.configuration.model, robot.configuration.data)
    robot.forward()

    skeleton = SkeletonRenderer(
        robot.configuration.model, args.width, args.height,
        args.lookat, args.elevation, args.azimuth, args.view_radius,
    )
    frame_w, frame_h = skeleton.frame_size()
    writer = FfmpegVideoWriter(args.output, frame_w, frame_h, args.fps)

    init_q = {"left": np.zeros(6), "right": np.zeros(6)}
    n_ok, n_fail = 0, 0

    for i, t in enumerate(transitions):
        state = np.asarray(t["observations"]["state"])
        if state.ndim > 1:
            state = state[-1]

        for side in ("left", "right"):
            pose6 = state[state_index[f"{side}/tcp_pose"]]
            target_pose = euler_pose_to_scalar_first_quat_pose(pose6)
            qpos = robot.compute_ik(target_pose, init_q[side], side=side)
            if qpos is None:
                n_fail += 1
                continue
            robot.configuration.update_qpos(qpos, side=side)
            init_q[side] = qpos
            n_ok += 1

        robot.forward()
        frame = skeleton.render(robot.configuration.data, title=f"step {args.start + i}")
        writer.write(frame)

    writer.release()
    print(f"wrote {len(transitions)} frames to {args.output} (ik ok={n_ok}, ik fail={n_fail})")


if __name__ == "__main__":
    main()
