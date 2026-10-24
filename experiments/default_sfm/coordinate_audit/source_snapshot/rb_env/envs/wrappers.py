import time
import threading
import queue
from importlib.resources import files

import gymnasium as gym
import numpy as np
import cv2
from rb_env.spacemouse.spacemouse_expert_udevfix import SpaceMouseExpert
from gymnasium import Env, spaces
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R
from rb_env.envs.action_adapter import MultiSideActionAdapter
from rb_env.envs.action_utils import (resolve_side_slices, world_delta_to_action, rpy_to_pose, pose_to_rpy)
from rb_env.envs.gripper_wrapper import GripperRobotWrapper
from rb_env.envs.visualizer import Visualizer
from rb_env.gripper_drivers import compute_noop_gripper_action
from rb_env.utils.scipy_spatial_utils import (
    apply_local_pose_offset_rotvec,
    calculate_global_pose_offset_rotvec,
    pose_to_transform,
    transform_to_pose
)

sigmoid = lambda x: 1 / (1 + np.exp(-x))

from collections import deque, defaultdict
def time_decorator(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        out = func(self, *args, **kwargs)
        time_elapsed = time.time() - start_time
        
        if not hasattr(self, "_timings"):
            self._timings = defaultdict(list)
        self._timings[func.__name__].append(time_elapsed)
        
        return out

    return wrapper



def state_history_quat_to_euler(state_history):
    state_history = np.array(state_history, copy=True, dtype=np.float32)
    if state_history.ndim != 2 or state_history.shape[-1] < 20:
        return state_history

    euler = R.from_quat(state_history[:, 7:11], scalar_first=True).as_euler("xyz")
    history_euler = np.concatenate(
        [
            state_history[:, :4],
            state_history[:, 4:7],
            euler.astype(np.float32),
            state_history[:, 11:],
        ],
        axis=-1,
    )
    return history_euler.astype(np.float32, copy=False)


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    @time_decorator
    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:], scalar_first=True).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class SidePrefixedQuat2EulerWrapper(gym.ObservationWrapper):
    """Convert TCP quaternions for every side present in the observation."""

    def __init__(self, env: Env):
        super().__init__(env)
        state_space = env.observation_space["state"]
        self.sides = tuple(
            side for side in ("left", "right") if f"{side}/tcp_pose" in state_space.spaces
        )
        if not self.sides:
            raise ValueError("No side-prefixed TCP pose found in observation space")

        for side in self.sides:
            pose_key = f"{side}/tcp_pose"
            if state_space[pose_key].shape != (7,):
                raise ValueError(f"{pose_key} must have shape (7,)")
            state_space[pose_key] = spaces.Box(-np.inf, np.inf, shape=(6,))

            history_key = f"{side}/state_history"
            if history_key in state_space.spaces:
                history_len = state_space[history_key].shape[0]
                state_space[history_key] = spaces.Box(
                    -np.inf, np.inf, shape=(history_len, 19)
                )
    
    @time_decorator
    def observation(self, observation):
        # convert tcp pose from quat to euler
        state = observation["state"]
        for side in self.sides:
            pose_key = f"{side}/tcp_pose"
            tcp_pose = state[pose_key]
            state[pose_key] = np.concatenate(
                (
                    tcp_pose[:3],
                    R.from_quat(tcp_pose[3:], scalar_first=True).as_euler("xyz"),
                )
            )
            history_key = f"{side}/state_history"
            if history_key in state:
                state[history_key] = state_history_quat_to_euler(state[history_key])

        return observation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info



def _identity_ft_to_tcp():
    return {
        "p_ft_tcp": np.zeros(3, dtype=np.float32),
        "r_ft_tcp": R.identity(),
        "source": "identity",
    }


def _parse_ft_to_tcp(value):
    """Parse T_ft_tcp: TCP origin/orientation expressed in the FT frame."""
    if value is None:
        return None

    p_ft_tcp = np.asarray(
        value.get("p_ft_tcp", value.get("translation", [0.0, 0.0, 0.0])),
        dtype=np.float32,
    )
    if p_ft_tcp.shape != (3,):
        raise ValueError(f"p_ft_tcp must have shape (3,), got {p_ft_tcp.shape}")

    rotation = value.get("r_ft_tcp")
    if isinstance(rotation, R):
        r_ft_tcp = rotation
    elif rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape == (3, 3):
            r_ft_tcp = R.from_matrix(rotation)
        elif rotation.shape == (4,):
            r_ft_tcp = R.from_quat(rotation, scalar_first=True)
        else:
            raise ValueError(
                "r_ft_tcp must be a scipy Rotation, a 3x3 matrix, or a WXYZ quaternion"
            )
    else:
        quat = value.get(
            "q_ft_tcp", value.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])
        )
        quat = np.asarray(quat, dtype=np.float64)
        if quat.shape != (4,):
            raise ValueError(f"q_ft_tcp must have shape (4,), got {quat.shape}")
        r_ft_tcp = R.from_quat(quat, scalar_first=True)

    return {
        "p_ft_tcp": p_ft_tcp,
        "r_ft_tcp": r_ft_tcp,
        "source": "provided",
    }


def _find_robot_nodes(env):
    base_env = env.unwrapped
    nodes = {}
    for side in ("left", "right"):
        side_env = getattr(base_env, f"env_{side}", None)
        nodes[side] = getattr(side_env, "robot_node", None)
    return nodes


class DualWrenchFrameWrapper(gym.Wrapper):
    """Express FT-sensor wrenches at the URDF-defined TCP in the base frame.

    ``tcp_pose`` already describes the final TCP and ``tcp_vel`` already comes from
    the controller in the base frame, so this wrapper deliberately leaves pose,
    twist, and actions unchanged.  A static transform can be supplied per side; if
    omitted, ``T_ft_tcp`` is resolved from TF and cached after the first success.
    """

    def __init__(
        self,
        env: Env,
        ft_to_tcp=None,
        tf_lookup_timeout=2.0,
        tf_retry_interval=1.0,
        tf_frame_prefix="dual_rb3",
    ):
        super().__init__(env)
        state_space = env.observation_space["state"]
        print('=================================')
        print(state_space)
        active_sides = tuple(
            side for side in ("left", "right") if f"{side}/tcp_pose" in state_space.spaces
        )
        if not active_sides:
            raise ValueError("No side-prefixed TCP pose found in observation space")
        for side in active_sides:
            if state_space[f"{side}/tcp_pose"].shape != (7,):
                raise ValueError(f"{side}/tcp_pose must have shape (7,)")

        ft_to_tcp = ft_to_tcp or {}
        self._ft_to_tcp = {
            side: _parse_ft_to_tcp(ft_to_tcp.get(side))
            for side in ("left", "right")
        }
        self._identity_transform = _identity_ft_to_tcp()
        self._tf_lookup_timeout = float(tf_lookup_timeout)
        self._tf_retry_interval = float(tf_retry_interval)
        self._tf_frame_prefix = (tf_frame_prefix or "").strip("/")
        self._robot_nodes = _find_robot_nodes(env)
        self._last_tf_attempt = {"left": float("-inf"), "right": float("-inf")}
        self._tf_warned = set()

    def _frame(self, side, link):
        suffix = f"{side}/{link}"
        if not self._tf_frame_prefix:
            return suffix
        return f"{self._tf_frame_prefix}/{suffix}"

    def _warn_tf_fallback(self, side, message):
        if side in self._tf_warned:
            return
        node = self._robot_nodes[side]
        if node is not None and hasattr(node, "get_logger"):
            node.get_logger().warning(message)
        self._tf_warned.add(side)

    def _resolve_ft_to_tcp(self, side):
        cached = self._ft_to_tcp[side]
        if cached is not None:
            return cached

        node = self._robot_nodes[side]
        if node is None or not hasattr(node, "ros_tf_handler"):
            self._warn_tf_fallback(
                side,
                f"Using identity FT-to-TCP transform for {side}; no ROS TF handler is available",
            )
            return self._identity_transform

        now = time.monotonic()
        if now - self._last_tf_attempt[side] < self._tf_retry_interval:
            return self._identity_transform
        self._last_tf_attempt[side] = now

        try:
            from rclpy.duration import Duration
            from rclpy.time import Time

            transform = node.ros_tf_handler.tf_buffer.lookup_transform(
                self._frame(side, "ft_sensor_link"),
                self._frame(side, "tcp"),
                Time(),
                Duration(seconds=self._tf_lookup_timeout),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            cached = {
                "p_ft_tcp": np.array(
                    [translation.x, translation.y, translation.z], dtype=np.float32
                ),
                "r_ft_tcp": R.from_quat(
                    [rotation.w, rotation.x, rotation.y, rotation.z],
                    scalar_first=True,
                ),
                "source": "tf",
            }
            self._ft_to_tcp[side] = cached
            return cached
        except Exception as exc:
            self._warn_tf_fallback(
                side,
                f"Using identity FT-to-TCP transform for {side}; TF lookup failed: {exc}",
            )
            return self._identity_transform

    @staticmethod
    def _wrench_to_base_at_tcp(force_ft, torque_ft, tcp_quat_wxyz, ft_to_tcp):
        force_ft = np.asarray(force_ft, dtype=np.float32)
        torque_ft = np.asarray(torque_ft, dtype=np.float32)
        p_ft_tcp = ft_to_tcp["p_ft_tcp"]
        r_tcp_ft = ft_to_tcp["r_ft_tcp"].inv()

        # Shift the moment from the FT origin to the TCP origin, then express it
        # in TCP coordinates: tau_tcp = R_tcp_ft (tau_ft - p_ft_tcp x F_ft).
        force_tcp = r_tcp_ft.apply(force_ft)
        torque_tcp = r_tcp_ft.apply(torque_ft - np.cross(p_ft_tcp, force_ft))

        r_base_tcp = R.from_quat(tcp_quat_wxyz, scalar_first=True)
        return (
            r_base_tcp.apply(force_tcp).astype(np.float32),
            r_base_tcp.apply(torque_tcp).astype(np.float32),
        )

    def _transform_state_history(self, state_history, ft_to_tcp):
        state_history = np.array(state_history, copy=True, dtype=np.float32)
        if state_history.ndim != 2 or state_history.shape[-1] < 20:
            return state_history

        force, torque = self._wrench_to_base_at_tcp(
            state_history[:, 1:4],
            state_history[:, 11:14],
            state_history[:, 7:11],
            ft_to_tcp,
        )
        state_history[:, 1:4] = force
        state_history[:, 11:14] = torque
        return state_history.astype(np.float32, copy=False)

    @time_decorator
    def observation(self, observation):
        state = observation["state"]
        for side in ("left", "right"):
            tcp_pose_key = f"{side}/tcp_pose"
            if tcp_pose_key not in state:
                continue

            force_key = f"{side}/tcp_force"
            torque_key = f"{side}/tcp_torque"
            wrench_history_key = f"{side}/wrench_history"
            state_history_key = f"{side}/state_history"
            has_current_wrench = force_key in state and torque_key in state
            if not (
                has_current_wrench
                or wrench_history_key in state
                or state_history_key in state
            ):
                continue

            # Resolve at most once per side and reuse for every field in this sample.
            ft_to_tcp = self._resolve_ft_to_tcp(side)
            tcp_quat = state[tcp_pose_key][3:]

            if has_current_wrench:
                force, torque = self._wrench_to_base_at_tcp(
                    state[force_key], state[torque_key], tcp_quat, ft_to_tcp
                )
                state[force_key] = force
                state[torque_key] = torque

            if wrench_history_key in state:
                wrench_history = state[wrench_history_key]
                # This history contains no synchronized poses, so its samples use
                # the current TCP orientation.  state_history below carries and
                # uses a per-sample quaternion.
                history_force, history_torque = self._wrench_to_base_at_tcp(
                    wrench_history[:, :3],
                    wrench_history[:, 3:6],
                    tcp_quat,
                    ft_to_tcp,
                )
                state[wrench_history_key] = np.concatenate(
                    (history_force, history_torque), axis=-1
                )

            if state_history_key in state:
                state[state_history_key] = self._transform_state_history(
                    state[state_history_key], ft_to_tcp
                )

        return observation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        return self.observation(obs), rew, done, truncated, info



class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    @time_decorator
    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    @time_decorator
    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def symlog(v, temperature=1.0):
    log_value = np.log(1.0+np.abs(v)*temperature) / np.log(1.0+temperature)
    return np.sign(v) * log_value


def invsymlog(v, temperature=1.0):
    v = np.asarray(v, dtype=float)
    exp_value = np.expm1(np.log1p(temperature) * np.abs(v)) / temperature
    return np.sign(v) * exp_value


def validate_spacemouse_symlog_temperature(value):
    """Validate before opening SpaceMouse/robot devices."""
    try:
        temperature = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("spacemouse_symlog_temperature must be finite and > 0") from None
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("spacemouse_symlog_temperature must be finite and > 0")
    return temperature


class DualSpacemouseIntervention2(gym.ActionWrapper):
    def __init__(
        self,
        env,
        spacemouse_action_scales,
        action_indices=None,
        gripper_enabled=True,
        arm_type="dual",
        spacemouse_symlog_temperatures=None,
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.spacemouse_action_scales = {
            side: self._validate_spacemouse_action_scale(side, scale)
            for side, scale in spacemouse_action_scales.items()
        }

        temperatures = spacemouse_symlog_temperatures or {}
        self.spacemouse_symlog_temperatures = {
            side: validate_spacemouse_symlog_temperature(temperatures.get(side, 10.0))
            for side in self.spacemouse_action_scales
        }
        self.expert = SpaceMouseExpert(active_sides=tuple(self.spacemouse_action_scales))
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

        self.left1_last_toggle_time = 0
        self.right1_last_toggle_time = 0
        self.left2_last_toggle_time = 0
        self.right2_last_toggle_time = 0
        self.debounce_interval = (
            0.5  # Minimum time interval between toggles (in seconds)
        )

        self.fix_orientation_button1 = False
        self.fix_orientation_button2 = False
        self.use_log_action = True
        self.arm_type = arm_type
        self.spacemouse_rotation_locked = False

    def toggle_spacemouse_rotation_lock(self):
        self.spacemouse_rotation_locked = not self.spacemouse_rotation_locked
        status = "locked (translation only)" if self.spacemouse_rotation_locked else "unlocked"
        print(f"[SpaceMouse] Rotation {status}")

    @time_decorator
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        expert_a = np.asarray(expert_a).copy()
        if self.spacemouse_rotation_locked:
            expert_a[3:6] = 0
            expert_a[9:12] = 0
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)

        current_time = time.time()
        if self.left2:
            if current_time - self.left2_last_toggle_time > self.debounce_interval:
                if self.fix_orientation_button1:
                    self.fix_orientation_button1 = False
                else:
                    self.fix_orientation_button1 = True
            self.left2_last_toggle_time = current_time

        if self.right2:
            if current_time - self.left2_last_toggle_time > self.debounce_interval:
                if self.fix_orientation_button2:
                    self.fix_orientation_button2 = False
                else:
                    self.fix_orientation_button2 = True
            self.left2_last_toggle_time = current_time

        if self.fix_orientation_button1:
            expert_a[0:3] = [0.0, 0.0, 0.0]
        if self.fix_orientation_button2:
            expert_a[6:9] = [0.0, 0.0, 0.0]

        if self.gripper_enabled:
            left_gripper_action = np.zeros((1,))
            right_gripper_action = np.zeros((1,))

            if (
                "left" in self.spacemouse_action_scales
                and current_time - self.left1_last_toggle_time > self.debounce_interval
            ):
                if self.left1:
                    if self.env_left.is_gripper_closed:
                        left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                    else:
                        left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                    intervened = True
                    self.left1_last_toggle_time = current_time
            if (
                "right" in self.spacemouse_action_scales
                and current_time - self.right1_last_toggle_time > self.debounce_interval
            ):
                if self.right1:
                    if self.env_right.is_gripper_closed:
                        right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                    else:
                        right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                    intervened = True

                    self.right1_last_toggle_time = current_time

            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            if self.use_log_action:
                expert_a = self._shape_spacemouse_action(expert_a)
            expert_a = self._to_si_action(expert_a)
            block_dim = 7 if self.gripper_enabled else 6
            if self.arm_type == "left":
                expert_a = expert_a[:block_dim]
            elif self.arm_type == "right":
                expert_a = expert_a[block_dim : 2 * block_dim]
            return world_delta_to_action(self.env, expert_a), True
        return action, False

    def _shape_spacemouse_action(self, action):
        clipped = np.clip(action, -1, 1)
        # Preserve the existing gripper button shaping; temperature controls
        # only the six motion axes of each active arm.
        shaped = invsymlog(clipped, temperature=10.0)
        block_dim = 7 if self.gripper_enabled else 6
        for side, index in (("left", 0), ("right", 1)):
            if side in self.spacemouse_symlog_temperatures:
                axes = slice(index * block_dim, index * block_dim + 6)
                shaped[axes] = invsymlog(
                    clipped[axes], temperature=self.spacemouse_symlog_temperatures[side]
                )
        return shaped

    @staticmethod
    def _validate_spacemouse_action_scale(side, scale):
        scale = np.asarray(scale, dtype=np.float32)
        if scale.shape == (2,):
            scale = np.concatenate(
                [np.repeat(scale[0], 3), np.repeat(scale[1], 3)]
            )
        elif scale.shape != (6,):
            raise ValueError(
                f"{side} spacemouse_action_scale must contain either 2 values "
                "[translation_m, rotation_rad] or 6 axis values "
                f"[x_m, y_m, z_m, rx_rad, ry_rad, rz_rad], got {scale}"
            )
        if not np.all(np.isfinite(scale)) or np.any(scale < 0):
            raise ValueError(
                f"{side} spacemouse_action_scale must contain finite, "
                f"non-negative values, got {scale}"
            )
        return scale

    def _to_si_action(self, action):
        """Convert SpaceMouse motion to SI and gripper input to configured units."""
        action = np.asarray(action, dtype=np.float32).copy()
        block_dim = 7 if self.gripper_enabled else 6
        for side, block_index in (("left", 0), ("right", 1)):
            if side not in self.spacemouse_action_scales:
                continue
            offset = block_index * block_dim
            action[offset : offset + 6] *= self.spacemouse_action_scales[side]
            if self.gripper_enabled:
                arm = getattr(self.unwrapped, f"env_{side}")
                close_m = getattr(arm, "gripper_binary_close_threshold_m", None)
                if arm.gripper_action_mode == "binary" and close_m is not None:
                    open_m = arm.gripper_binary_open_threshold_m
                    gripper = action[offset + 6]
                    if gripper <= -0.05:
                        action[offset + 6] = close_m
                    elif gripper >= 0.05:
                        action[offset + 6] = open_m
                    else:
                        action[offset + 6] = (close_m + open_m) / 2.0
                elif arm.gripper_action_mode == "continuous":
                    out_lo, out_hi = arm.gripper_continuous_output_range
                    gripper = action[offset + 6]
                    if gripper <= -0.05:
                        target_pct = out_lo
                    elif gripper >= 0.05:
                        target_pct = out_hi
                    else:
                        target_pct = arm._last_gripper_target
                        if target_pct is None:
                            target_pct = arm.curr_gripper_pos * 100.0
                    action[offset + 6] = compute_noop_gripper_action(
                        {
                            "mode": arm.gripper_action_mode,
                            "input_range": arm.gripper_continuous_input_range,
                            "output_range": arm.gripper_continuous_output_range,
                        },
                        target=target_pct,
                    )
        return action

    @time_decorator
    def step(self, action):
        new_action, replaced = self.action(action)
        # print(new_action, replaced)  액션프린트
        obs, rew, done, truncated, info = self.env.step(new_action.copy())
        if replaced:
            info["intervene_action"] = new_action.copy()
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.expert.close()
        return self.env.close()


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    @time_decorator
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    @time_decorator
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info


class DualRelativeActionWrapper(gym.ActionWrapper):
    def __init__(
            self, env, *,
            use_each_gripper: bool = True, 
            use_arm_grip: bool = False, 
            arm_grip_offset: float = 0.02, 
            xyz_clip_low = (None, None, None),
            xyz_clip_high = (None, None, None)
        ):
        super().__init__(env)
        self.use_each_gripper = use_each_gripper
        self.use_arm_grip = use_arm_grip
        self.arm_grip_offset = arm_grip_offset
        self.xyz_clip_low = xyz_clip_low
        self.xyz_clip_high = xyz_clip_high

        if self.use_arm_grip:
            self.observation_space['state']['gripper_pose'] = gym.spaces.Box(-1, 1, (1,))
            self.is_gripper_closed = False
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )
        self.reset_left_tcp_transform = None
        self.reset_right_tcp_transform = None
    

    @time_decorator
    def action(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()  # mutable!

        self.reference_pose = apply_local_pose_offset_rotvec(
            self.reference_pose, action[:-1]
        )
        self.reference_pose[:3] = np.clip(self.reference_pose[:3], self.xyz_clip_low, self.xyz_clip_high)
        reference_transform = pose_to_transform(self.reference_pose)

        left_target_pose = transform_to_pose(reference_transform @ self.reset_left_tcp_transform)
        right_target_pose = transform_to_pose(reference_transform @ self.reset_right_tcp_transform)

        if not self.use_each_gripper:
            if action[-1] < -0.5:
                self.is_gripper_closed = True
            elif action[-1] > 0.5:
                self.is_gripper_closed = False
            
            if self.use_arm_grip and self.is_gripper_closed:
                left_target_pose[1] -= self.arm_grip_offset
                right_target_pose[1] += self.arm_grip_offset


        left_offset = calculate_global_pose_offset_rotvec(
            self.env_left.currpos, left_target_pose
        )
        right_offset = calculate_global_pose_offset_rotvec(
            self.env_right.currpos, right_target_pose
        )

        if self.use_each_gripper:
            new_action = np.concatenate([left_offset, action[-1], right_offset, action[-1]])
        else:
            new_action = np.concatenate([left_offset, right_offset])
        return new_action

    @time_decorator
    def step(self, action):
        obs, rew, term, trun, info = self.env.step(self.action(action))
        obs['state']['gripper_pose'] = np.array((self.is_gripper_closed,), dtype=np.float32)
        return obs, rew, term, trun, info

    def reset(self, **kwargs):
        self.is_gripper_closed = False
        obs, info = self.env.reset(**kwargs)

        self.reset_left_tcp_transform = pose_to_transform(self.env_left.currpos)
        self.reset_right_tcp_transform = pose_to_transform(self.env_right.currpos)

        self.reference_pose = [0, 0, 0, 1, 0, 0, 0]

        obs['state']['gripper_pose'] = np.array((self.is_gripper_closed,), dtype=np.float32)
        return obs, info
    


class ActionAdapterWrapper(gym.Wrapper):
    """Adapt the world-frame error of an absolute target, then rebuild its pose.

    The calibration map still operates on XYZ/rotvec deltas. Only this boundary
    converts those deltas; the wrapped environment always receives XYZ/RPY.
    Human intervention stays inside this wrapper and bypasses adaptation.
    """

    DELTA_CHUNK_MODE = "delta"

    def __init__(
        self,
        env,
        adapters: MultiSideActionAdapter,
        *,
        sides=("left", "right"),
        chunk_mode: str = DELTA_CHUNK_MODE,
        enabled: bool = True,
        log_info: bool = True,
    ):
        super().__init__(env)
        if chunk_mode != self.DELTA_CHUNK_MODE:
            raise ValueError(
                "ActionAdapterWrapper only applies to "
                f"action_chunk_mode {self.DELTA_CHUNK_MODE!r}, got {chunk_mode!r}. "
                "Disable action_adapter.enable, or switch the chunk mode."
            )

        self.enabled = bool(enabled)
        self.adapters = adapters
        self.sides = tuple(sides)
        self.log_info = bool(log_info)
        self.action_dim = int(env.action_space.shape[0])

        base_env = env.unwrapped
        self._slices = {}
        self._side_envs = {}
        for side in self.sides:
            if side not in adapters:
                raise KeyError(
                    f"no action adapter for side {side!r}; "
                    f"available: {adapters.sides}"
                )
            self._slices[side] = resolve_side_slices(self.action_dim, side)
            side_env = getattr(base_env, f"env_{side}", None)
            if side_env is None:
                raise ValueError(
                    f"side {side!r} is not available in the environment"
                )
            self._side_envs[side] = side_env

    # -- action ----------------------------------------------------------

    @time_decorator
    def action(self, action):
        """The command to send for a desired world-frame delta action."""
        return self._adapt(action)[0]

    def _adapt(self, action):
        """Return ``(action_to_send, {side: (target_delta, command)})``."""
        if not self.enabled:
            return action, {}

        adapted = np.asarray(action)
        if adapted.shape != (self.action_dim,):
            raise ValueError(
                f"Expected an action of shape ({self.action_dim},), got "
                f"{adapted.shape}"
            )
        if not np.issubdtype(adapted.dtype, np.floating):
            adapted = adapted.astype(np.float32)
        else:
            adapted = adapted.copy()

        pending = {}
        for side in self.sides:
            pos_slice, rot_slice, _ = self._slices[side]
            current = np.asarray(self._side_envs[side].currpos, dtype=float)
            target = rpy_to_pose(adapted[pos_slice.start:rot_slice.stop])
            target_delta = calculate_global_pose_offset_rotvec(current, target)
            command = self.adapters[side].predict(target_delta)
            from rb_env.utils.scipy_spatial_utils import apply_global_pose_offset_rotvec
            adapted[pos_slice.start:rot_slice.stop] = pose_to_rpy(
                apply_global_pose_offset_rotvec(current, command))
            pending[side] = (target_delta, command)
        return adapted, pending

    # -- gym api ---------------------------------------------------------

    @time_decorator
    def step(self, action):
        adapted, pending = self._adapt(action)
        poses_before = {
            side: np.asarray(self._side_envs[side].currpos, dtype=float).copy()
            for side in pending
        }

        obs, reward, terminated, truncated, info = self.env.step(adapted)

        if pending:
            diagnostics = self._adapt_online(pending, poses_before, info)
            if self.log_info:
                info["action_adapter"] = diagnostics
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        for _, adapter in self.adapters.items():
            if adapter.online.reset_each_episode:
                adapter.reset_online()
        return self.env.reset(**kwargs)

    # -- online adaptation -----------------------------------------------

    def _adapt_online(self, pending, poses_before, info):
        # A human took over this step, so the arm's motion says nothing about
        # how the controller answers *this adapter's* commands.
        intervened = (
            "intervene_action" in info or "action_intervene" in info
        )
        diagnostics = {}
        for side, (target_delta, command) in pending.items():
            observed_delta = calculate_global_pose_offset_rotvec(
                poses_before[side],
                np.asarray(self._side_envs[side].currpos, dtype=float),
            )
            error = (
                None
                if intervened
                else self.adapters[side].update(observed_delta, command)
            )
            diagnostics[side] = {
                "target_delta": target_delta,
                "command": command,
                "observed_delta": observed_delta,
                "error_norm": None if error is None else float(np.linalg.norm(error)),
                "updated": error is not None,
            }
        return diagnostics


class ImageDisplayer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self.state_viz = Visualizer(str(files("rb_env.envs").joinpath("gripper.stl")))
        self.queue = queue.Queue(maxsize=1)
        self.display_scale = 1.3

        ### Force 시각화 ###
        self.start_time = time.time()
        self.force_times = deque(maxlen=2000)
        self.force_left = deque(maxlen=2000)
        self.force_right = deque(maxlen=2000)

        self.force_window_sec = 10.0  

        self.FORCE_W = 256
        self.FORCE_GRAPH_H = 256 
        self.FORCE_TOP_MARGIN = 24    # arm_label 전용 영역
        self.FORCE_H = self.FORCE_GRAPH_H + self.FORCE_TOP_MARGIN

        self.force_img_left = np.zeros((self.FORCE_H, self.FORCE_W, 3), dtype=np.uint8)
        self.force_img_right = np.zeros((self.FORCE_H, self.FORCE_W, 3), dtype=np.uint8)
        self.prev_force_left = None
        self.prev_force_right = None

        self.torque_img_left = np.zeros((self.FORCE_H, self.FORCE_W, 3), dtype=np.uint8)
        self.torque_img_right = np.zeros((self.FORCE_H, self.FORCE_W, 3), dtype=np.uint8)
        self.prev_torque_left = None
        self.prev_torque_right = None
        self.runner = threading.Thread(
            target=self._display_loop,
            name="observation-display",
            daemon=True,
        )
        self.runner.start()
    

    @time_decorator
    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self._enqueue_latest(self._display_payload(obs))
        return obs, reward, done, trunc, info

    @staticmethod
    def _display_payload(obs):
        """Snapshot only the small state values; image arrays stay read-only."""
        state = obs["state"]

        def vector(key):
            return np.asarray(
                state.get(key, np.zeros(3)), dtype=np.float32
            ).copy()

        return (
            tuple(obs["images"].values()),
            vector("left/tcp_force"),
            vector("right/tcp_force"),
            vector("left/tcp_torque"),
            vector("right/tcp_torque"),
        )

    def _enqueue_latest(self, payload):
        """Publish without waiting; discard a stale frame if display lags."""
        try:
            self.queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            # The display thread raced us and another producer filled the slot.
            pass

    def close(self):
        self._enqueue_latest(None)
        self.runner.join(timeout=1.0)
        return self.env.close()

    def _display_loop(self):
        while True:
            payload = self.queue.get()
            if payload is None:
                return
            frame = self._render_frame(payload)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.display_scale != 1.0:
                height, width = bgr_frame.shape[:2]
                bgr_frame = cv2.resize(
                    bgr_frame,
                    (
                        int(width * self.display_scale),
                        int(height * self.display_scale),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow("Image", bgr_frame)
            cv2.waitKey(1)

    def _render_frame(self, payload):
        images, left_force, right_force, left_torque, right_torque = payload
        if images:
            height = max(image.shape[0] for image in images)
            # Cameras can have different native resolutions. Pad only the
            # preview; observation and recording arrays retain their pixels.
            row1 = np.concatenate([
                cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)
                if image.shape[0] != height else image
                for image in images
            ], axis=1)
        else:
            row1 = np.zeros((256, 256, 3), dtype=np.uint8)

        target_size = (self.FORCE_W, self.FORCE_H)
        now = time.time() - self.start_time

        self.force_times.append(now)
        self.force_left.append(left_force)
        self.force_right.append(right_force)

        self._update_vector_image(
            self.force_img_left,
            self.prev_force_left, left_force,
            arm_label="Left force",
            axis_names=("Fx", "Fy", "Fz"),
            scale=1.0,
            f_min=-20.0, f_max=20.0,
        )
        self._update_vector_image(
            self.force_img_right,
            self.prev_force_right, right_force,
            arm_label="Right force",
            axis_names=("Fx", "Fy", "Fz"),
            scale=1.0,
            f_min=-20.0, f_max=20.0,
        )
        self.prev_force_left = left_force
        self.prev_force_right = right_force

        force_img_left = cv2.resize(self.force_img_left, target_size)
        force_img_right = cv2.resize(self.force_img_right, target_size)
        force_row = np.concatenate([force_img_left, force_img_right], axis=1)

        self._update_vector_image(
            self.torque_img_left,
            self.prev_torque_left, left_torque,
            arm_label="Left torque",
            axis_names=("Tx", "Ty", "Tz"),
            scale=20.0, 
            f_min=-20.0, f_max=20.0 # 필요하면 여기도 조절 가능
        )
        self._update_vector_image(
            self.torque_img_right,
            self.prev_torque_right, right_torque,
            arm_label="Right torque",
            axis_names=("Tx", "Ty", "Tz"),
            scale=20.0,
            f_min=-20.0, f_max=20.0
        )
        self.prev_torque_left = left_torque
        self.prev_torque_right = right_torque

        torque_img_left = cv2.resize(self.torque_img_left, target_size)
        torque_img_right = cv2.resize(self.torque_img_right, target_size)
        torque_row = np.concatenate([torque_img_left, torque_img_right], axis=1)

        target_w = row1.shape[1]

        def pad_or_resize(row, target_w):
            if row.shape[1] < target_w:
                pad_w = target_w - row.shape[1]
                zero_pad = np.zeros((row.shape[0], pad_w, 3), dtype=row.dtype)
                return np.concatenate([row, zero_pad], axis=1)
            elif row.shape[1] > target_w:
                return cv2.resize(row, (target_w, row.shape[0]))
            else:
                return row

        force_row = pad_or_resize(force_row, target_w)
        torque_row = pad_or_resize(torque_row, target_w)
        return np.concatenate([row1, force_row, torque_row], axis=0)
    
    def _update_vector_image(self, img, prev_vec, new_vec,
                             f_min=-20.0, f_max=20.0,
                             arm_label=None,
                             axis_names=("Fx", "Fy", "Fz"),
                             scale=1.0):
        """
        img: (H, W, 3) uint8 image buffer (in-place)
        prev_vec, new_vec: (3,)  -> force or torque vector
        axis_names: 각 밴드에 표시할 라벨 ("Fx","Fy","Fz") or ("Tx","Ty","Tz")
        scale: 값에 곱해줄 스케일 (torque 확대용)
        """
        H, W, _ = img.shape
        LABEL_W = 60             # 왼쪽 축 라벨 영역 폭

        # --- 새로 추가된 부분: 위쪽 margin과 그래프 영역 분리 ---
        H_top = getattr(self, "FORCE_TOP_MARGIN", 0)   # 위쪽 라벨 영역 높이
        H_graph = H - H_top                             # 실제 그래프 그릴 영역 높이
        H_each = H_graph // 3                           # 밴드 height

        # 1) 그래프 부분만 왼쪽으로 스크롤
        img[:, LABEL_W:-1] = img[:, LABEL_W+1:]
        img[:, -1] = 0

        # 2) 라벨 영역 깨끗하게 비우기 (왼쪽 + 위쪽 모두 초기화)
        img[:, :LABEL_W] = 0

        if H_top > 0:
            img[0:H_top, LABEL_W:] = 0

        # 3) 값 → y좌표 맵핑 (scale 적용)
        def val_to_y(i, v):
            v_scaled = v * scale
            v_clamped = max(min(v_scaled, f_max), f_min)
            ratio = (v_clamped - f_min) / (f_max - f_min + 1e-6)
            top_band = H_top + i * H_each
            bottom_band = top_band + H_each - 1
            return int(bottom_band - ratio * (H_each - 1))

        x_new = W - 1
        x_prev = W - 2

        if prev_vec is None:
            prev_vec = new_vec

        # 4) 원점(0선) 먼저 그리기
        zero_color = (255, 0, 0)  # red in BGR
        for i in range(3):
            y0 = val_to_y(i, 0.0)
            cv2.line(img, (LABEL_W, y0), (W - 1, y0), zero_color, 1)

        # 5) 우측 상단 arm label (위쪽 margin 안에만 그리기)
        if arm_label is not None and H_top > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (tw, th), baseline = cv2.getTextSize(arm_label, font,
                                                 font_scale, thickness)
            # 라벨을 위쪽 margin 안에 넣기
            x_text = W - tw - 5
            # margin 안에서 vertical center
            y_text = max(th + 2, H_top // 2 + th // 2)

            y0_rect = 0
            y1_rect = H_top - 1

            cv2.rectangle(
                img,
                (x_text - 2, y0_rect),
                (W, y1_rect),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                img,
                arm_label,
                (x_text, min(y_text, y1_rect - baseline)),
                font,
                font_scale,
                (200, 200, 200),
                thickness,
                cv2.LINE_AA
            )

        # 6) 왼쪽 라벨 + 흰색 그래프 선
        line_color = (255, 255, 255)  # white (BGR)
        for i, axis_name in enumerate(axis_names):
            base_y = H_top + i * H_each

            # 왼쪽 축 라벨 (Fx/Fy/Fz 또는 Tx/Ty/Tz)
            cv2.putText(
                img,
                axis_name,
                (5, base_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150, 150, 255),
                1,
                cv2.LINE_AA
            )

            # 현재 값 표시 (소수점 둘째 자리)
            val_disp = float(new_vec[i])   # WORLD 값 그대로
            val_text = f"{val_disp:.2f}"
            cv2.putText(
                img,
                val_text,
                (5, base_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )

            # 그래프 선
            y_prev = val_to_y(i, float(prev_vec[i]))
            y_new = val_to_y(i, float(new_vec[i]))
            cv2.line(img, (x_prev, y_prev), (x_new, y_new), line_color, 1)
