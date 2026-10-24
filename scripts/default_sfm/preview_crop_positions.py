"""Preview historical USB crop bounds on the first frame of each task/position.

This reads the first two image buffers without unpickling the entire demo.
The extraction was checked byte-for-byte against conventional pickle loading
of both cameras in the USB demo used by the preprocessing audit. No training
data is generated or changed.
"""

from __future__ import annotations

import json
import pickletools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "experiments/default_sfm/preprocessing_audit"
CROPS = {"head_cam_d455": (96, 128, 288, 384), "wrist_cam_d405_a": (72, 122, 336, 448)}
IMAGE_BYTES = 480 * 640 * 3


def first_images(path):
    found = {}
    camera = None
    with path.open("rb") as stream:
        for opcode, value, _ in pickletools.genops(stream):
            if isinstance(value, str) and value in CROPS:
                camera = value
            if opcode.name in ("BINBYTES", "BINBYTES8", "BYTEARRAY8") and len(value) == IMAGE_BYTES:
                if camera is None or camera in found:
                    raise ValueError(f"Unexpected first-frame camera layout: {path}")
                found[camera] = np.frombuffer(value, dtype=np.uint8).reshape(480, 640, 3)
                if len(found) == len(CROPS):
                    return found, stream.tell()
    raise ValueError(f"Missing first-frame camera buffers: {path}")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    inventory = json.loads((ROOT / "experiments/default_sfm/setup_records/source_inventory.json").read_text())
    selected = {}
    for entry in sorted(inventory["files"], key=lambda entry: entry["path"]):
        if entry["path"].endswith(".pkl"):
            selected.setdefault(entry["path"].split("/")[0], entry["path"])
    assert len(selected) == 45
    directory = AUDIT / "vision/positions"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for folder, relative in selected.items():
        images, read_bytes = first_images(Path("/data/crestlux/default_sfm") / relative)
        for camera, pixels in images.items():
            Image.fromarray(pixels).save(directory / f"{folder}_{camera}.png")
        records.append({"folder": folder, "source": relative, "frame_index": 0, "bytes_read": read_bytes})
    tasks = sorted({folder.rsplit("_p", 1)[0] for folder in selected})
    for camera, (top, left, height, width) in CROPS.items():
        fig, axes = plt.subplots(len(tasks), 5, figsize=(15, 22.5), dpi=100)
        fig.subplots_adjust(left=.01, right=.99, top=.96, bottom=.025, hspace=.15, wspace=.025)
        for row, task in enumerate(tasks):
            for col in range(5):
                folder = f"{task}_p{col + 1}"
                ax = axes[row, col]
                ax.imshow(Image.open(directory / f"{folder}_{camera}.png"))
                ax.add_patch(Rectangle((left, top), width, height, fill=False, edgecolor="#ffcc33", linewidth=1.5))
                ax.set_title(folder, fontsize=10, pad=3)
                ax.set_axis_off()
        fig.suptitle(f"{camera}: historical USB ROI on native 640x480 images", fontsize=17, y=.986)
        fig.text(.5, .008, "One demo per task/position, first frame only. This does not verify the full trajectories.", ha="center", fontsize=11)
        fig.savefig(directory.parent / f"all45_{camera}_crop_reference.png")
        plt.close(fig)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one first frame from one demo for each of 45 task/position folders",
        "does_not_verify_full_trajectory_crop_coverage": True,
        "extraction": "pickletools first named camera byte buffers; verified against full-load USB reference images",
        "crop_reference_tlhw": CROPS,
        "records": records,
        "training_preprocessing_applied": False,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved 90 original first-frame images and two crop-reference contact sheets.")


if __name__ == "__main__":
    main()
