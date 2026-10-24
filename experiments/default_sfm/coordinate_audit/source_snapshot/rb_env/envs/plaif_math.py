"""ROS-independent pose conversions for the PLAIF controller adapter."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_from_wxyz(quaternion) -> Rotation:
    """Create a SciPy rotation without requiring its newer scalar_first API."""

    wxyz = np.asarray(quaternion, dtype=float)
    if wxyz.shape != (4,):
        raise ValueError("quaternion must contain 4 values [qw,qx,qy,qz]")
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]])


def rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    """Return ``[qw,qx,qy,qz]`` on both old and new SciPy releases."""

    xyzw = np.asarray(rotation.as_quat(), dtype=float)
    return xyzw[[3, 0, 1, 2]]


def normalized_pose(pose) -> np.ndarray:
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("pose must be 7 finite values [x,y,z,qw,qx,qy,qz]")
    norm = np.linalg.norm(value[3:])
    if norm < 1e-12:
        raise ValueError("pose quaternion must be non-zero")
    result = value.copy()
    result[3:] /= norm
    return result


def absolute_target_to_flange_delta(current_pose, target_pose) -> np.ndarray:
    """Convert base-frame TCP poses to a flange-frame translation/rotvec delta.

    Both poses already use the controller's native, flange-aligned TCP axes.
    No fixed axis rotation or TCP-to-flange origin shift is applied. This is
    the follow wire's absolute-to-incremental conversion, not a tool correction.
    """

    current = normalized_pose(current_pose)
    target = normalized_pose(target_pose)
    base_from_flange = rotation_from_wxyz(current[3:])
    translation = base_from_flange.inv().apply(target[:3] - current[:3])
    rotation = (
        base_from_flange.inv()
        * rotation_from_wxyz(target[3:])
    ).as_rotvec()
    return np.concatenate((translation, rotation))


def bounded_target(current, target, translation_limit, rotation_limit):
    delta = target[:3] - current[:3]
    distance = np.linalg.norm(delta)
    xyz = current[:3] + delta * min(1., translation_limit / max(distance, 1e-15))
    rc = rotation_from_wxyz(current[3:])
    rv = (rc.inv()*rotation_from_wxyz(target[3:])).as_rotvec()
    rv *= min(1., rotation_limit / max(np.linalg.norm(rv), 1e-15))
    return np.concatenate((xyz, rotation_to_wxyz(rc*Rotation.from_rotvec(rv))))



def absolute_targets_to_flange_deltas(current_pose, targets, *,
                                      max_translation_step=None, max_rotation_step=None):
    """Encode consecutive absolute targets, advancing by each bounded wire delta."""
    current = normalized_pose(current_pose)
    result = []
    for target in targets:
        delta = absolute_target_to_flange_delta(current, target)
        for part, limit in ((slice(0, 3), max_translation_step),
                            (slice(3, 6), max_rotation_step)):
            if limit is not None:
                delta[part] *= min(1., limit / max(np.linalg.norm(delta[part]), 1e-15))
        result.append(delta)
        rotation = rotation_from_wxyz(current[3:])
        current = np.concatenate((
            current[:3] + rotation.apply(delta[:3]),
            rotation_to_wxyz(rotation * Rotation.from_rotvec(delta[3:])),
        ))
    return np.asarray(result)
