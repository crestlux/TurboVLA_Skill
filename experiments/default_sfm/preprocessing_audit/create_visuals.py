"""Deterministic crop and Euler-angle audit figures from recorded observations."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(), "axes.unicode_minus": False})
CROPS = {"head_cam_d455": (96, 128, 288, 384), "wrist_cam_d405_a": (72, 122, 336, 448)}


def letterbox224(image):
    content = cv2.resize(image, (224, 168), interpolation=cv2.INTER_AREA)
    result = np.zeros((224, 224, 3), dtype=np.uint8)
    result[28:196] = content
    return result


def vision_figure():
    directory = ROOT / "vision"
    record = json.loads((directory / "preview_source.json").read_text())
    frame = record["peak_force_frame"]
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.1), dpi=120, facecolor="#f7f8fa")
    fig.subplots_adjust(left=.03, right=.985, top=.84, bottom=.10, wspace=.09, hspace=.24)
    titles = ["원본 640×480 + USB crop 영역", "Jack FAFLOW 경로 재현 · 224×224", "원본에서 직접 crop · 224×224"]
    transforms = {}
    for row, (camera, crop) in enumerate(CROPS.items()):
        raw = np.asarray(Image.open(directory / f"usb_p1_f{frame:03d}_{camera}_raw.png"))
        top, left, height, width = crop
        native = raw[top:top + height, left:left + width]
        reduced = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(reduced, (640, 480), interpolation=cv2.INTER_LINEAR)
        legacy = letterbox224(restored[top:top + height, left:left + width])
        direct = letterbox224(native)
        for name, pixels in (("native_crop", native), ("legacy_simulation_224", legacy), ("direct_crop_224", direct)):
            Image.fromarray(pixels).save(directory / f"{camera}_{name}.png")
        for col, pixels in enumerate((raw, legacy, direct)):
            ax = axes[row, col]
            ax.imshow(pixels, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            label = "HEAD" if row == 0 else "WRIST"
            ax.set_title(f"{label}  |  {titles[col]}", fontsize=11, pad=9)
        axes[row, 0].add_patch(Rectangle((left, top), width, height, fill=False, edgecolor="#ffcc33", linewidth=2.5))
        axes[row, 0].set_xlabel(f"x [{left}:{left+width}], y [{top}:{top+height}]  →  {width}×{height}", fontsize=11)
        transforms[camera] = {"crop_tlhw": list(crop), "crop_xyxy": [left, top, left+width, top+height],
            "native_crop_shape": list(native.shape), "original": f"usb_p1_f{frame:03d}_{camera}_raw.png"}
    fig.text(.03, .95, "TurboVLA USB 모델의 crop 영역과 원본 crop 비교", fontsize=22, weight="bold", color="#172b49")
    fig.text(.03, .90, f"현재 default_sfm / usb_p1 / frame {frame}의 실제 사진 · 노란 사각형: csilab의 TurboVLA USB 번들에서 확인한 영역", fontsize=12, color="#4d5868")
    fig.text(.5, .052, "같은 224×224 출력으로 비교: 가운데는 선축소·재확대 후 crop / 오른쪽은 640×480 원본에서 직접 crop", ha="center", fontsize=12)
    fig.text(.5, .022, "확정된 계획: 기존 USB 영역을 원본에서 crop → 224×224. 학습용 데이터 전처리는 아직 적용하지 않았습니다.", ha="center", fontsize=11, color="#5b6572")
    fig.savefig(directory / "usb_crop_comparison.png", dpi=120)
    fig.savefig(directory / "turbovla_usb_crop_comparison.png", dpi=120)
    plt.close(fig)
    (directory / "crop_evidence.json").write_text(json.dumps({
        "source": record, "reference_runtime": "/home/crestlux/project/faflow_final/preprocessing/runtime.py:12-24,36-51",
        "cameras": transforms, "comparison_output_shape": [224,224,3],
        "legacy_panel_is_simulation_on_current_raw_image": True,
        "turbovla_usb_source": "turbovla_usb_source_snapshot/jack_models_20261020/models/jack_faflow_delta_gru500hz/bundles/delta/inference_config.json",
        "turbovla_640_input_direct_crop_source": "turbovla_usb_source_snapshot/go_models_20261019/models/go_ft500_delta/bundles/delta/inference_config.json",
        "all_verified_turbovla_usb_bundle_sizes": [224, 224],
        "comparison_scope": "same current native image through the two bundled runtime recipes; not a claim about historical raw image resolution",
        "direct_pipeline": "native RGB -> crop -> resize224x168 INTER_AREA -> black padding28 top/bottom",
        "training_preprocessing_applied": False, "new_training_output_resolution_confirmed": True,
        "new_training_crop_bounds_confirmed": True,
    }, ensure_ascii=False, indent=2) + "\n")


def rpy_example():
    report = json.loads((ROOT / "rpy/report.json").read_text())
    row = report["demos"][0]
    numeric = np.load(ROOT / "rpy" / row["numeric_artifact"])
    state = numeric["state"].astype(float)
    t = numeric["time"] - numeric["time"][0]
    angles = state[:,7:10]
    mapped = np.remainder(angles, 2*np.pi)
    fig, axes = plt.subplots(3,2,figsize=(12.6,8.2),dpi=120,facecolor="white")
    fig.subplots_adjust(left=.085,right=.98,top=.85,bottom=.12,wspace=.20,hspace=.30)
    for i, name in enumerate(("Roll", "Pitch", "Yaw")):
        for col, data in enumerate((angles, mapped)):
            axes[i,col].plot(t,data[:,i],lw=1,color="#bf483e" if col==0 else "#176b99")
            axes[i,col].set_ylim((-3.4,3.4) if col==0 else (-.2,6.5))
            axes[i,col].grid(alpha=.2)
            jumps=int(np.count_nonzero(np.abs(np.diff(data[:,i])) > np.pi))
            axes[i,col].text(.97,.90,f"큰 점프 {jumps}회",transform=axes[i,col].transAxes,ha="right",va="top",fontsize=11)
            if col==0:
                axes[i,col].set_ylabel(f"{name} (rad)")
            if i==2:
                axes[i,col].set_xlabel("데모 시작 후 시간 (s)")
    axes[0,0].set_title("현재 저장된 RPY",fontsize=14,pad=12)
    axes[0,1].set_title("모든 축에 θ mod 2π를 적용한 경우",fontsize=14,pad=12)
    fig.text(.085,.955,"0~2π 변환은 경계의 위치를 바꿉니다",fontsize=22,weight="bold",color="#172b49")
    fig.text(.085,.905,"실제 audio_jack_p1 데모 1개 · Roll/Yaw 점프는 사라지지만 Pitch에는 0↔2π 점프가 생깁니다",fontsize=12)
    fig.text(.5,.035,"큰 점프: 연속한 두 관측값의 차이가 π보다 큰 경우. 원본 파일은 수정하지 않았습니다.",ha="center",fontsize=11,color="#596373")
    fig.savefig(ROOT / "rpy/rpy_boundary_example.png",dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    vision_figure()
    rpy_example()
    print("Saved USB crop comparison and RPY boundary example")
