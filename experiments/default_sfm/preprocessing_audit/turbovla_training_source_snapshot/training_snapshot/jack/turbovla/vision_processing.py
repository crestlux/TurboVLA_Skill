from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_rgb_normalizer(
    image_mean: Sequence[float],
    image_std: Sequence[float],
    *,
    expected_size: int,
    size_divisor: int,
    backbone_name: str,
    rescale_factor: float = 1.0 / 255.0,
):
    """Build a resize-free RGB normalizer shared by training and evaluation."""

    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)

    def process_one(image: Image.Image | np.ndarray) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        width, height = image.size
        if (height, width) != (expected_size, expected_size):
            raise ValueError(
                f"{backbone_name} expects {expected_size}x{expected_size} RGB input, "
                f"but got {height}x{width}; this processor intentionally does not resize."
            )
        if height % size_divisor or width % size_divisor:
            raise ValueError(
                f"{backbone_name} input size {(height, width)} must be divisible by {size_divisor}."
            )
        array = np.asarray(image, dtype=np.float32) * float(rescale_factor)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - mean) / std

    def processor(images, return_tensors: str | None = None) -> dict[str, torch.Tensor]:
        del return_tensors
        if not isinstance(images, (list, tuple)):
            images = [images]
        return {"pixel_values": torch.stack([process_one(image) for image in images], dim=0)}

    return processor


def build_resnet50_processor(expected_size: int = 256, output_stride: int = 16):
    return build_rgb_normalizer(
        IMAGENET_MEAN,
        IMAGENET_STD,
        expected_size=expected_size,
        size_divisor=output_stride,
        backbone_name=f"ResNet-50-OS{output_stride}",
    )
