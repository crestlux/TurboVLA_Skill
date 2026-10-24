"""Episode training data made from transmitted commands, never unexecuted plans."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import pickle
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from rb_env.envs.action_utils import resolve_side_slices, rpy_to_pose, pose_to_rpy
from rb_env.utils.scipy_spatial_utils import (
    calculate_global_pose_offset_rotvec,
    apply_local_pose_offset_rotvec, apply_global_pose_offset_rotvec,
)


def encode_action_chunk(commands, reference_poses, gripper_start, *, mode,
                        yaw_deg=0.0, gripper_mode="absolute", rotation_unit="rad"):
    """Encode [H,S,7] absolute XYZ/RPY/gripper commands in policy coordinates.

    relative_tcp uses one fixed reference; delta/delta_flange use each step's
    measured pose, in global/local axes respectively. The caller supplies references.
    """
    commands = np.asarray(commands, dtype=float)
    references = np.asarray(reference_poses, dtype=float)
    if commands.ndim != 3 or commands.shape[-1] != 7 or references.shape != commands.shape:
        raise ValueError("commands and references must be [H, sides, 7]")
    if mode not in ("relative_tcp", "delta", "delta_flange") or gripper_mode not in ("absolute", "delta"):
        raise ValueError("unsupported action convention")
    if rotation_unit not in ("rad", "deg"):
        raise ValueError("rotation_unit must be rad or deg")
    result = commands.copy()
    correction = Rotation.from_euler("z", yaw_deg, degrees=True)
    for t in range(len(commands)):
        for side in range(commands.shape[1]):
            target = rpy_to_pose(commands[t, side, :6])
            reference = references[t, side]
            if mode in ("relative_tcp", "delta_flange"):
                base_rotation = Rotation.from_quat(reference[3:], scalar_first=True)
                target_rotation = Rotation.from_quat(target[3:], scalar_first=True)
                offset = np.r_[base_rotation.inv().apply(target[:3] - reference[:3]),
                               (base_rotation.inv() * target_rotation).as_rotvec()]
                if mode == "relative_tcp":
                    offset = np.r_[correction.inv().apply(offset[:3]),
                                   correction.inv().apply(offset[3:])]
            else:
                offset = calculate_global_pose_offset_rotvec(reference, target)
            result[t, side, :6] = offset
    if mode == "relative_tcp" and gripper_mode == "delta":
        result[:, :, 6] = np.diff(np.vstack((np.asarray(gripper_start)[None], commands[:, :, 6])), axis=0)
    if rotation_unit == "deg":
        result[:, :, 3:6] = np.rad2deg(result[:, :, 3:6])
    return result.reshape(len(commands), -1)


def decode_action_chunk(actions, reference_poses, gripper_start, *, mode,
                        yaw_deg=0.0, gripper_mode="absolute", rotation_unit="rad"):
    """Recover exact recorded absolute targets; bypass online path tracking.

    This verifies command equivalence, not identical physical robot motion.
    """
    references = np.asarray(reference_poses, dtype=float)
    if mode not in ("relative_tcp", "delta", "delta_flange"):
        raise ValueError("unsupported action convention")
    values = np.asarray(actions, dtype=float).reshape(references.shape).copy()
    if rotation_unit == "deg":
        values[:, :, 3:6] = np.deg2rad(values[:, :, 3:6])
    correction = Rotation.from_euler("z", yaw_deg, degrees=True)
    result = values.copy()
    for t in range(len(values)):
        for side in range(values.shape[1]):
            offset = values[t, side, :6]
            if mode in ("relative_tcp", "delta_flange"):
                if mode == "relative_tcp":
                    offset = np.r_[correction.apply(offset[:3]), correction.apply(offset[3:])]
                target = apply_local_pose_offset_rotvec(references[t, side], offset)
            else:
                target = apply_global_pose_offset_rotvec(references[t, side], offset)
            result[t, side, :6] = pose_to_rpy(target)
    if mode == "relative_tcp" and gripper_mode == "delta":
        result[:, :, 6] = np.asarray(gripper_start)[None] + np.cumsum(values[:, :, 6], axis=0)
    return result


class EpisodeRecorder:
    def __init__(self, env, initial_observation, *, config, prompt, policy_sides,
                 mode, yaw_deg, gripper_mode, rotation_unit):
        self.env = env
        self.sides = tuple(policy_sides)
        self.action_dim = int(env.action_space.shape[0])
        self.convention = dict(mode=mode, yaw_deg=float(yaw_deg),
                               gripper_mode=gripper_mode, rotation_unit=rotation_unit)
        self.episode_id = uuid.uuid4().hex
        self.metadata = dict(config=copy.deepcopy(config), prompt=prompt,
                             policy_sides=self.sides, action_convention=self.convention,
                             action_period_s=1.0 / float(config["train"].get("hz", 10)),
                             command_format="absolute_tcp_xyz_rpy_gripper",
                             chunk_source="first_transmitted_target_per_control_step")
        self.transitions = []
        self.previous_obs = copy.deepcopy(initial_observation)

    def begin_step(self):
        poses, grippers, observations = [], [], {}
        for side in self.sides:
            arm = getattr(self.env.unwrapped, f"env_{side}")
            sample = getattr(arm, 'curr_gripper_observation', None)
            if isinstance(sample, dict):
                observations[side] = copy.deepcopy(sample)
            poses.append(np.asarray(arm.currpos, dtype=float).copy())
            in_lo, in_hi = arm.gripper_continuous_input_range
            out_lo, out_hi = arm.gripper_continuous_output_range
            grippers.append(float(np.clip(in_lo + (float(arm.curr_gripper_pos) * 100 - out_lo)
                                           / (out_hi - out_lo) * (in_hi - in_lo), in_lo, in_hi)))
        return dict(poses=np.stack(poses), grippers=np.asarray(grippers),
                    gripper_observations=observations,
                    timestamp=time.time(), monotonic_time=time.monotonic())

    def append(self, context, requested_action, next_obs, reward, terminated, truncated, info, horizon):
        # Missing receipts must never silently turn raw model plans into labels.
        if "executed_action" not in info:
            raise RuntimeError("Environment must report executed_action to record training episodes")
        executed = np.asarray(info["executed_action"], dtype=float)
        if (executed.ndim != 2 or not len(executed) or executed.shape[1] != self.action_dim
                or not np.all(np.isfinite(executed))):
            raise ValueError("executed_action must be finite [H, action_dim]")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("server chunk horizon must be a positive integer")
        observation = copy.deepcopy(next_obs)
        self.transitions.append(dict(
            observations=self.previous_obs,
            # actions is the first actual target; the full short wire chunk is separate.
            actions=executed[0].copy(), executed_action_chunk=executed.copy(),
            requested_action=np.asarray(requested_action, dtype=float).copy(),
            rewards=float(reward), terminated=bool(terminated), truncated=bool(truncated),
            dones=bool(terminated or truncated), masks=1.0 - bool(terminated),
            infos=copy.deepcopy(info), action_context=context, policy_horizon=horizon,
            intervened="intervene_action" in info or "action_intervene" in info,
        ))
        self.previous_obs = observation

    def finish(self, result):
        if result["outcome"] not in ("success", "failure"):
            return None
        if not self.transitions or not (self.transitions[-1]["terminated"] or self.transitions[-1]["truncated"]):
            raise ValueError("Only completed episodes can be uploaded for training")
        for index, transition in enumerate(self.transitions):
            horizon = transition["policy_horizon"]
            future = self.transitions[index:index + horizon]
            count = len(future)
            commands = np.empty((count, len(self.sides), 7), dtype=float)
            references = np.empty_like(commands)
            for t, following in enumerate(future):
                for side_index, side in enumerate(self.sides):
                    pos, _, grip = resolve_side_slices(self.action_dim, side)
                    commands[t, side_index] = following["actions"][pos.start:grip + 1]
                references[t] = (transition if self.convention["mode"] == "relative_tcp"
                                 else following)["action_context"]["poses"]
            encoded = encode_action_chunk(commands, references,
                                           transition["action_context"]["grippers"], **self.convention)
            padded = np.zeros((horizon, encoded.shape[1]), dtype=float)
            padded[:count] = encoded
            mask = np.arange(horizon) < count
            transition["action_chunk"] = padded
            transition["action_chunk_mask"] = mask
            transition["action_chunk_reference_poses"] = references
            transition["action_chunk_gripper_start"] = transition["action_context"]["grippers"].copy()
        return dict(schema_version=1, episode_id=self.episode_id,
                    result=copy.deepcopy(result), metadata=self.metadata,
                    num_steps=len(self.transitions),
                    episode_return=sum(t["rewards"] for t in self.transitions),
                    # Intermediate next observations are the following transition's
                    # observations; only the terminal observation needs its own field.
                    transitions=self.transitions, final_observation=self.previous_obs)


def save_episode(episode, directory):
    """Keep an atomic local copy before any network operation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{episode['episode_id']}.pkl"
    temporary = destination.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(episode, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination
