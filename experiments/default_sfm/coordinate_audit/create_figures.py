"""Plot recorded photographs beside measured TCP coordinates; never edit raw demos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(), "axes.unicode_minus": False})
STATE = np.load(ROOT / "audio_jack_p1_trajectory.npz")["state"]


def make_figure(axis: str, before: int, after: int, title: str, color: str) -> dict:
    fig = plt.figure(figsize=(14, 7.25), dpi=120, facecolor="#f7f8fa")
    fig.text(0.04, 0.945, title, fontsize=22, weight="bold", color=color)
    fig.text(
        0.04, 0.900,
        "audio_jack_p1 실제 수집 사진 · 고정 head 카메라 · 화면 왼쪽 로봇팔의 그리퍼를 비교하세요",
        fontsize=12, color="#4a5464",
    )
    records = []
    for left, frame, label in ((0.04, before, "이동 전"), (0.53, after, "이동 후")):
        path = ROOT / f"audio_jack_p1_f{frame:03d}_head_cam_d455.png"
        rgb = np.asarray(Image.open(path))
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        # Only layout and captions are drawn. The photograph has no arrows,
        # cropping, mirroring, retouching, or generated pixels.
        ax = fig.add_axes([left, 0.20, 0.43, 0.623])
        ax.imshow(rgb, interpolation="none")
        ax.set_axis_off()
        fig.text(left, 0.846, f"{label}  |  frame {frame}", fontsize=14, weight="bold", color="#1d2735")
        xyz = STATE[frame, 4:7]
        fig.text(
            left, 0.161, f"x = {xyz[0]:.4f} m     y = {xyz[1]:.4f} m     z = {xyz[2]:.4f} m",
            fontsize=13, color="#1d2735",
        )
        records.append({"frame": frame, "xyz_m": xyz.tolist(), "image": path.name,
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    fig.text(0.50, 0.51, "→", ha="center", va="center", fontsize=34, color=color)
    delta = (STATE[after, 4:7].astype(float) - STATE[before, 4:7].astype(float)) * 1000
    fig.text(0.5, 0.085, f"실제 변화량    Δx = {delta[0]:+.1f} mm     Δy = {delta[1]:+.1f} mm     Δz = {delta[2]:+.1f} mm",
             ha="center", fontsize=17, weight="bold", color=color)
    fig.text(0.5, 0.030, "원본 PKL의 observations에서 추출한 사진과 state[..., 4:7]의 실측값", ha="center", fontsize=11, color="#5c6572")
    path = ROOT / f"{axis}_direction.png"
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"axis": axis, "frames": records, "delta_xyz_mm": delta.tolist(), "figure": path.name}


def main() -> None:
    figures = [
        make_figure("x", 30, 60, "+X 증가: 작업대 앞쪽으로 — 이 사진에서는 아래쪽", "#b83c2f"),
        make_figure("y", 14, 30, "+Y 증가: 작업대 옆 방향으로 — 이 사진에서는 오른쪽", "#087b60"),
    ]
    evidence = {
        "source_demo": "/data/crestlux/default_sfm/audio_jack_p1/2026-10-19/1_demos_2026-10-19_17-40-21_330892.pkl",
        "camera": "head_cam_d455",
        "pose_reference_frame": "cell base frame",
        "state_indices_zero_based": {"tcp_x": 4, "tcp_y": 5, "tcp_z": 6},
        "positive_directions_in_these_head_camera_images": {
            "x": "Toward the front edge of the worktable; downward, with perspective-dependent lateral motion",
            "y": "Toward image right",
        },
        "direction_evidence": "Measured state changes paired with actual fixed-camera observations; no camera calibration or pixel-to-meter transform was fitted",
        "source_code_evidence": {
            "base_frame": "source_snapshot/rb_env/envs/plaif_node.py:3-7",
            "xyz_callback": "source_snapshot/rb_env/envs/plaif_node.py:281-300",
            "xyz_preserved_by_orientation_conversion": "source_snapshot/rb_env/envs/wrappers.py:114-121",
            "state_flattening": "source_snapshot/rb_env/envs/serl_obs_wrappers.py:63-68",
        },
        "position_unit": "meter (ROS Pose convention; no scale conversion in the observed state path)",
        "unit_reference": "https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-0103.rst",
        "figures": figures,
        "raw_demo_modified": False,
        "training_preprocessing_applied": False,
        "training_started": False,
    }
    manifest = json.loads((ROOT.parent / "setup_records/copy_manifest.json").read_text())
    relative_demo = Path(evidence["source_demo"]).relative_to("/data/crestlux/default_sfm").as_posix()
    evidence["source_demo_sha256"] = next(row["sha256"] for row in manifest["files"] if row["path"] == relative_demo)
    (ROOT / "coordinate_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": [row["figure"] for row in figures], "evidence": "coordinate_evidence.json"}))


if __name__ == "__main__":
    main()
