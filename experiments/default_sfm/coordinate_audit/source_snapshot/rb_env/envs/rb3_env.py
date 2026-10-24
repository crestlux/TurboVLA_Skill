"""Gym Interface for Franka"""

import copy
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict

import cv2
import gymnasium as gym
import numpy as np
from rb_env.utils.scipy_spatial_utils import rpy_to_quaternion
from rb_env.utils.rich_utils import print_panel
from scipy.spatial.transform import Rotation

from rb_env.envs.action_utils import rpy_to_pose, pose_to_rpy
from rb_env.gripper_drivers import GripperControlConfig
from rb_env.utils.timer import Stopwatch


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if len(img_array) == 0:  # None is our signal to exit
                break
            frame = np.concatenate(
                [
                    cv2.resize(v, (512, 512))
                    for k, v in img_array.items()
                    if "full" not in k
                ],
                axis=1,
            )
            try:
                cv2.imshow(self.name, frame)
                cv2.waitKey(33)
            except Exception as e:
                print(e)
        else:
            print("Image Displayer Exitted!")


##############################################################################


@dataclass
class EnvConfig:
    """Validated per-arm values consumed by :class:`Rb3Env`.

    Pose arrays use ``[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]``.
    Environment actions are absolute cell-base XYZ/RPY plus a gripper value.
    """

    ROBOT_PREFIX: str
    RESET_POSE: np.ndarray
    RESET_POSE_MAX: np.ndarray | None
    ABS_POSE_LIMIT_LOW: np.ndarray
    ABS_POSE_LIMIT_HIGH: np.ndarray
    MAX_EPISODE_LENGTH: int
    DISPLAY_IMAGE: bool
    GRIPPER_CONTROL: Dict[str, Any] = field(default_factory=dict)
    CAMERAS: Dict[str, Any] = field(default_factory=dict)
    IMAGE_CROP: Dict[str, Callable] = field(default_factory=dict)
    TARGET_POSE: np.ndarray = field(default_factory=lambda: np.zeros(6))
    GRASP_POSE: np.ndarray = field(default_factory=lambda: np.zeros(6))
    REWARD_THRESHOLD: np.ndarray = field(default_factory=lambda: np.zeros(6))
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: tuple[float, ...] = (0.0,)
    RANDOM_RZ_RANGE: tuple[float, ...] = (0.0,)
    COMPLIANCE_PARAM: Dict[str, float] = field(default_factory=dict)
    RESET_PARAM: Dict[str, float] = field(default_factory=dict)
    PRECISION_PARAM: Dict[str, float] = field(default_factory=dict)
    LOAD_PARAM: Dict[str, Any] = field(
        default_factory=lambda: {
            "mass": 0.0,
            "F_x_center_load": [0.0, 0.0, 0.0],
            "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    GRIPPER_SLEEP: float = 0.6
    JOINT_RESET_PERIOD: int = 0
    MAX_TRANSLATION_SPEED_MPS: float = 0.05
    MAX_ROTATION_SPEED_RADPS: float = 0.1

    def __post_init__(self):
        for name in ("MAX_TRANSLATION_SPEED_MPS", "MAX_ROTATION_SPEED_RADPS"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)


# Kept as a source-compatible name for downstream code.
DefaultEnvConfig = EnvConfig


##############################################################################

# pure_delta_env=True,

class Rb3Env(gym.Env):
    RESET_TRANSLATION_SPEED_MPS = 0.06
    MIN_RESET_TARGET_DURATION = 0.5
    RESET_REACH_TIMEOUT = 2.0

    def __init__(
        self,
        robot_node,
        hz=10,
        fake_env=False,
        config: EnvConfig | None = None,
        set_load=False,
    ):
        self.fake_env = fake_env
        self.robot_node = robot_node

        self._TARGET_POSE = config.TARGET_POSE
        self._reset_pose = config.RESET_POSE
        if config.RESET_POSE_MAX is None:
            self._reset_pose_max = self._reset_pose
        else:
            self._reset_pose_max = config.RESET_POSE_MAX
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.max_translation_speed_mps = config.MAX_TRANSLATION_SPEED_MPS
        self.max_rotation_speed_radps = config.MAX_ROTATION_SPEED_RADPS
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.robot_prefix = self.config.ROBOT_PREFIX
        if hasattr(self.robot_node, "get_gripper_control_config"):
            gripper_control = self.robot_node.get_gripper_control_config(
                self.robot_prefix, config.GRIPPER_CONTROL
            )
        else:
            gripper_control = GripperControlConfig().with_overrides(
                config.GRIPPER_CONTROL
            )
        if gripper_control.mode not in {"binary", "continuous"}:
            raise ValueError(
                f"Unsupported gripper action mode: {gripper_control.mode!r}"
            )
        if gripper_control.input_range[0] == gripper_control.input_range[1]:
            raise ValueError("gripper input_range endpoints must differ")

        self.gripper_action_mode = gripper_control.mode
        from rb_env.gripper_interface import GripperInterface
        interface_getter = getattr(self.robot_node, 'get_gripper_interface', None)
        interface = interface_getter(self.robot_prefix, config.GRIPPER_CONTROL) if callable(interface_getter) else None
        self.gripper_interface = interface if isinstance(interface, GripperInterface) else GripperInterface(gripper_control)
        self.gripper_binary_close_threshold_m = gripper_control.binary_close_threshold_m
        self.gripper_binary_open_threshold_m = gripper_control.binary_open_threshold_m
        self.gripper_binary_closed_target = None
        self.gripper_binary_open_target = None
        if gripper_control.mode == "binary" and gripper_control.binary_closed_width_m is not None:
            self.gripper_binary_closed_target = self.robot_node.gripper_width_to_percent(
                self.robot_prefix, gripper_control.binary_closed_width_m
            )
            self.gripper_binary_open_target = self.robot_node.gripper_width_to_percent(
                self.robot_prefix, gripper_control.binary_open_width_m
            )
        self.gripper_continuous_input_range = gripper_control.input_range
        self.gripper_continuous_output_range = gripper_control.output_range
        self.gripper_continuous_force = gripper_control.force
        self.gripper_continuous_vel = gripper_control.vel
        self.gripper_cmd_deadband = gripper_control.cmd_deadband
        self.gripper_closed_eps = gripper_control.closed_eps
        self._last_gripper_target = None
        self.curr_gripper_telemetry = {}
        self.is_gripper_closed = False
        # self.pure_delta_env = pure_delta_env

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], rpy_to_quaternion(config.RESET_POSE[3:])]
        )
        self._update_currpos()
        self.lastsent = time.time()
        self.hz = hz
        self._step_deadline = None
        self.history_len = getattr(self.robot_node, "history_len", 10)
        self.state_history_dim = getattr(self.robot_node, "state_history_dim", 20)
        self.joint_reset_cycle = (
            config.JOINT_RESET_PERIOD
        )  # reset the robot joint every 200 cycles

        
        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.array([-np.inf] * 6 + [self.gripper_continuous_input_range[0]], dtype=np.float32),
            np.array([np.inf] * 6 + [self.gripper_continuous_input_range[1]], dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "wrench_history": gym.spaces.Box(
                            -np.inf, np.inf, shape=(self.history_len, 6)
                        ),
                        "state_history": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(self.history_len, self.state_history_dim),
                        ),
                    }
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self._last_step_time = time.time()
        self.is_gripper_moved = False
        self.is_reward_ok = True
        self.force_limit_cnt = 0

        print("Initialized Franka")
        print("xyz bounding box:", self.xyz_bounding_box)
        print("rpy bounding box:", self.rpy_bounding_box)

        # reset 값 확인
        print(f"[{self.robot_prefix}] _reset_pose     :", self._reset_pose)
        print(f"[{self.robot_prefix}] _reset_pose_max :", self._reset_pose_max)


    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""

        # clip if current pose is out of bounds
        clipped_cur_pos = np.clip(self.currpos[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        clipped_targ_pos = np.clip(pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        pose[:3] = np.where(clipped_cur_pos == self.currpos[:3], pose[:3], clipped_targ_pos)
        
        return pose

    def step(self, action: np.ndarray) -> tuple:
        """Track absolute [x,y,z,roll,pitch,yaw,gripper] for one control cycle.

        A target is a reference, not a promise of arrival within this period.
        """
        action = np.asarray(action, dtype=float)
        is_chunk = action.ndim == 2
        if (not (action.shape == (7,) or
                 (is_chunk and action.shape[1:] == (7,) and 1 <= len(action) <= 50))
                or not np.all(np.isfinite(action))):
            raise ValueError("action must be finite absolute XYZ/RPY + gripper: [7] or [H, 7], H=1..50")
        if is_chunk and getattr(self.robot_node, "supports_action_chunks", False) is not True:
            raise ValueError("action chunks require a PLAIF backend")
        self._update_currpos()
        if is_chunk:
            targets = np.stack([self.clip_safety_box(rpy_to_pose(row[:6])) for row in action])
            self.nextpos = targets[0].copy()
        else:
            self.nextpos = self.clip_safety_box(rpy_to_pose(action[:6]))
            targets = self.nextpos
        command_start_pose = self.currpos.copy()
        target_duration = 1.0 / self.hz
        cycle_started = time.monotonic()

        gripper_action = action[0, -1] if is_chunk else action[-1]
        self._send_gripper_command(gripper_action, mode=self.gripper_action_mode)
        target_wrench = np.zeros(6)
        sent_command = self._send_pos_command(
            targets,
            target_wrench,
            target_duration=target_duration,
            start_pose=command_start_pose,
        )

        if getattr(self, 'episode_clock_paused', False) is not True:
            self.curr_path_length += 1
        # Keep a monotonic schedule across steps. Camera/recording/wrapper
        # work happens after this wait and before the next command; starting
        # a fresh full period on every call adds that cost to every interval.
        # A 103 ms sender feeding the 100 ms PLAIF consumer periodically leaves
        # an empty playback slot, which brakes the stream even without a stall.
        period = 1.0 / self.hz
        previous_deadline = getattr(self, "_step_deadline", None)
        if previous_deadline is None or cycle_started >= previous_deadline + period:
            deadline = cycle_started + period
        else:
            deadline = previous_deadline + period
        finished = time.monotonic()
        # Rebase on an overrun or long interruption; never replay missed steps.
        self._step_deadline = max(deadline, finished)
        time.sleep(max(0.0, deadline - finished))

        self._update_currpos()
        ob = self._get_obs()

        reward = 0.0
        done = bool(
            self.curr_path_length >= self.max_episode_length
            or reward > 0
            or self.terminate
        )

        infos = self.get_infos()
        infos["succeed"] = reward
        sent_poses = (sent_command["target_poses"] if isinstance(sent_command, dict)
                      else np.atleast_2d(targets))
        sent_gripper = float(gripper_action)
        gripper_target = getattr(self, "_last_gripper_target", None)
        if self.gripper_action_mode == "continuous" and gripper_target is not None:
            from rb_env.gripper_drivers import compute_noop_gripper_action
            sent_gripper = compute_noop_gripper_action({
                "mode": "continuous",
                "input_range": self.gripper_continuous_input_range,
                "output_range": self.gripper_continuous_output_range,
            }, target=gripper_target)
        infos["executed_action"] = np.stack([
            np.r_[pose_to_rpy(pose), sent_gripper] for pose in sent_poses
        ])
        infos["command_receipt"] = sent_command if isinstance(sent_command, dict) else None
        infos["gripper_target_percent"] = gripper_target

        self._last_step_time = time.time()


        # truncated is always False
        
        return ob, reward, done, False, infos

    def get_infos(self):
        infos = {}
        infos["current_robot_timestamp"] = self.curr_robot_timestamp
        if hasattr(self, "curr_camera_timestamp"):
            infos["current_camera_timestamp"] = self.curr_camera_timestamp
        else:
            infos["current_camera_timestamp"] = None
        infos["is_reward_ok"] = self.is_reward_ok
        infos["current_qpos"] = self.q
        infos["current_gripper_telemetry"] = self.curr_gripper_telemetry
        sample = getattr(self, 'curr_gripper_observation', None)
        if isinstance(sample, dict):
            infos['gripper_observation'] = dict(sample)

        return infos

    def interpolate_move(
        self,
        goal: np.ndarray,
        target_duration: float,
        reach_timeout: float = RESET_REACH_TIMEOUT,
    ):
        """Schedule one interpolated target and wait for tracking to settle."""
        if goal.shape == (6,):
            goal_position = goal[:3]
            goal_orientation = rpy_to_quaternion(goal[3:])
        else:
            goal_position = goal[:3]
            goal_orientation = goal[3:]

        self._update_currpos()
        goal_pose = np.concatenate([goal_position, goal_orientation])
        move_linear = getattr(getattr(self, "robot_node", None), "move_linear", None)
        if callable(move_linear):
            distance = np.linalg.norm(goal_pose[:3] - self.currpos[:3])
            speed_mps = max(0.001, distance / max(target_duration, 1e-6))
            move_linear(
                goal_pose,
                side=self.robot_prefix,
                speed_mps=speed_mps,
                # target_duration estimates translation only. Controller MOVL
                # can take longer when rotation or joint limits dominate.
                timeout=max(30.0, target_duration + reach_timeout + 10.0),
            )
            self.nextpos = goal_pose
            self._update_currpos()
            return
        self._send_pos_command(
            goal_pose,
            is_clip=False,
            target_duration=target_duration,
        )

        interpolation_deadline = time.monotonic() + target_duration
        reach_deadline = interpolation_deadline + reach_timeout
        is_reached = False
        while time.monotonic() < reach_deadline:
            self._update_currpos()
            if time.monotonic() >= interpolation_deadline:
                is_reached = self.check_pose_reached(self.currpos, goal_pose)
                if is_reached:
                    break
            time.sleep(1.0 / self.hz)

        if not is_reached:
            logger = getattr(self.robot_node, "get_logger", lambda: None)()
            if logger is not None:
                logger.warning(
                    f"{self.robot_prefix} reset target was not reached within "
                    f"{target_duration + reach_timeout:.2f}s"
                )

        self.nextpos = goal_pose
        self._update_currpos()

    @classmethod
    def reset_target_duration(cls, distance_m: float) -> float:
        """Return reset command duration using the configured linear speed."""
        distance_m = float(distance_m)
        if not np.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("distance_m must be a finite non-negative value")
        return max(
            cls.MIN_RESET_TARGET_DURATION,
            distance_m / cls.RESET_TRANSLATION_SPEED_MPS,
        )

    def check_pose_reached(
        self,
        current_pose,
        target_pose,
        position_threshold=0.002,
        orientation_threshold=0.02,
    ):
        current_position = np.array(current_pose[:3])
        target_position = np.array(target_pose[:3])
        position_error = np.linalg.norm(current_position - target_position)

        current_orientation = Rotation.from_quat(current_pose[3:], scalar_first=True)
        target_orientation = Rotation.from_quat(target_pose[3:], scalar_first=True)
        orientation_error = np.abs(
            (target_orientation * current_orientation.inv()).as_rotvec()
        ).max()

        position_reached = position_error < position_threshold
        orientation_reached = orientation_error < orientation_threshold

        return position_reached and orientation_reached

    def go_to_reset(self, joint_reset=False, reset_pose=None):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        print('currpos', self.currpos)
        print('resetpos', self.resetpos)
        self._send_pos_command(self.currpos)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self.robot_node.dual_rb3_mujoco.trajectory_generator.plan_ptp(
                [self.currpos, self.resetpos],
                "pose",
                0.2,
                self.robot_prefix,
                init_q=self.q,
            )

        if reset_pose is None:
            reset_pos_xyz = np.random.uniform(
                self._reset_pose[:3], self._reset_pose_max[:3]
            )
        else:
            reset_pos_xyz = reset_pose[:3]

        reset_pos_euler = np.random.uniform(self._reset_pose[3:], self._reset_pose_max[3:])
        reset_pos_quat = rpy_to_quaternion(reset_pos_euler)
        reset_pose = np.concatenate([reset_pos_xyz, reset_pos_quat], axis=-1)
        distance = np.linalg.norm(self.currpos[:3] - reset_pose[:3], 2.0)
        self.interpolate_move(
            reset_pose,
            target_duration=self.reset_target_duration(distance),
        )

        # Change to compliance mode
        # requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, reset_pose=None, **kwargs):
        self._step_deadline = None
        if self.fake_env:
            raise NotImplementedError
        print_panel(f"{self.robot_prefix}_Rb3 Env Reset Start")
        
        self.robot_node.turn_cartesian_controller(self.robot_prefix)


        self.cycle_count += 1
        if (
            self.joint_reset_cycle != 0
            and self.cycle_count % self.joint_reset_cycle == 0
        ):
            self.cycle_count = 0
            joint_reset = True

        reset_gripper_target = getattr(self, "gripper_binary_open_target", None)
        if self.gripper_action_mode == "binary" and reset_gripper_target is not None:
            self.robot_node.control_gripper(
                length=reset_gripper_target,
                vel=self.gripper_continuous_vel,
                force=self.gripper_continuous_force,
                side=self.robot_prefix,
                is_wait_for_reached=False,
            )
        else:
            self.robot_node.open_gripper(self.robot_prefix, is_wait_for_reached=False)
            reset_gripper_target = 100.0

        self.go_to_reset(joint_reset=joint_reset ,reset_pose=reset_pose)
        self.curr_path_length = 0
        self.force_limit_cnt = 0
        self._last_gripper_target = reset_gripper_target
        self.is_gripper_closed = False

        self._update_currpos()
        self.nextpos = np.copy(self.currpos)
        obs = self._get_obs()
        self.terminate = False

        time.sleep(3.0)


        print_panel(f"{self.robot_prefix}_Rb3 Env Reset Done")
        print(f"Reset Position: {self.resetpos}")
        print(f"Current Position: {self.currpos}")
        return obs, {"succeed": False}


    def _send_pos_command(
        self,
        pos: np.ndarray,
        target_wrench: np.ndarray = np.zeros(6),
        is_clip=True,
        target_duration: float = 0.0,
        start_pose: np.ndarray | None = None,
    ):
        """Internal function to send position command to the robot."""
        if getattr(self.robot_node, "handles_delta_clipping", False) is True:
            # The backend computes and clips the delta from one live snapshot.
            # Do not pre-clip the target using a planned segment start.
            return self.robot_node.publish_compliance_target_frame(
                pos,
                target_wrench,
                self.robot_prefix,
                is_clip,
                target_duration=target_duration,
                max_translation_step=getattr(self, "max_translation_speed_mps", 0.05) / self.hz,
                max_rotation_step=getattr(self, "max_rotation_speed_radps", 0.1) / self.hz,
            )
        from rb_env.envs.plaif_math import bounded_target
        measured = np.asarray(self.currpos, dtype=float)
        pos = bounded_target(measured, pos,
                             getattr(self, "max_translation_speed_mps", 0.05) / self.hz,
                             getattr(self, "max_rotation_speed_radps", 0.1) / self.hz)
        start_pose = measured
        self.robot_node.publish_compliance_target_frame(
            pos,
            target_wrench,
            self.robot_prefix,
            is_clip,
            target_duration=target_duration,
            start_frame=start_pose,
        )
        return {"start_pose": start_pose.copy(), "target_poses": pos[None].copy()}


    def _send_gripper_command(self, pos: float, mode: str | None = None):
        """Internal function to send gripper command to the robot."""
        mode = mode or self.gripper_action_mode
        if mode == "binary":
            close_threshold = getattr(self, "gripper_binary_close_threshold_m", None)
            open_threshold = getattr(self, "gripper_binary_open_threshold_m", None)
            if close_threshold is None:
                close_threshold, open_threshold = -0.5, 0.5
            # Actions are float32; compare boundaries in the same representation.
            pos = np.float32(pos)
            if pos <= np.float32(close_threshold):
                closed = True
            elif pos >= np.float32(open_threshold):
                closed = False
            else:
                self.is_gripper_moved = False
                return
            target = getattr(self, "gripper_binary_closed_target" if closed
                             else "gripper_binary_open_target", None)
            same_state = closed == self.is_gripper_closed
            if same_state and (target is None or getattr(self, "_last_gripper_target", None) == target):
                self.is_gripper_moved = False
                return
            if target is not None:
                self.robot_node.control_gripper(
                    length=target,
                    vel=self.gripper_continuous_vel,
                    force=self.gripper_continuous_force,
                    side=self.robot_prefix,
                    is_wait_for_reached=False,
                )
            elif closed:
                self.robot_node.close_gripper(self.robot_prefix, is_wait_for_reached=False)
            else:
                self.robot_node.open_gripper(self.robot_prefix, is_wait_for_reached=False)
            self._last_gripper_target = target if target is not None else (0.0 if closed else 100.0)
            # Logical binary state: a nonzero configured closed width is still CLOSED.
            self.is_gripper_closed = closed
            self.is_gripper_moved = True
            return
        if mode != "continuous":
            raise ValueError(f"Unsupported gripper action mode: {mode!r}")

        in_lo, in_hi = self.gripper_continuous_input_range
        out_lo, out_hi = self.gripper_continuous_output_range
        pos = float(np.clip(pos, in_lo, in_hi))
        ratio = (pos - in_lo) / (in_hi - in_lo)
        target = out_lo + ratio * (out_hi - out_lo)

        if (
            self._last_gripper_target is not None
            and abs(target - self._last_gripper_target) < self.gripper_cmd_deadband
        ):
            self.is_gripper_moved = False
            return

        self._last_gripper_target = target
        self.robot_node.control_gripper(
            length=target,
            vel=self.gripper_continuous_vel,
            force=self.gripper_continuous_force,
            side=self.robot_prefix,
            is_wait_for_reached=False,
        )
        self.is_gripper_moved = False
        self.is_gripper_closed = target <= self.gripper_closed_eps * 100.0

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        if self.fake_env:
            return
        self.currpos = getattr(self.robot_node, f"current_{self.robot_prefix}_tcp_pose")
        self.currvel = getattr(self.robot_node, f"current_{self.robot_prefix}_tcp_twist")

        self.currforce = getattr(self.robot_node, f"current_{self.robot_prefix}_force")
        self.currtorque = getattr(self.robot_node, f"current_{self.robot_prefix}_torque")
        self.wrench_history = getattr(self.robot_node, f"history_{self.robot_prefix}_wrench")
        self.state_history = getattr(self.robot_node, f"history_{self.robot_prefix}_state")
        self.currjacobian = getattr(self.robot_node, f"current_{self.robot_prefix}_jacobian")

        self.q = getattr(self.robot_node, f"current_{self.robot_prefix}_qpos")
        self.dq = getattr(self.robot_node, f"current_{self.robot_prefix}_qvel")

        if hasattr(self.robot_node, "get_gripper_length"):
            self.curr_gripper_pos = (
                self.robot_node.get_gripper_length(self.robot_prefix) / 100
            )
            self.curr_gripper_telemetry = self.robot_node.get_gripper_telemetry(
                self.robot_prefix
            )
        else:
            self.curr_gripper_pos = (
                getattr(
                    self.robot_node,
                    f"current_{self.robot_prefix}_dh_grip_length",
                )
                / 100
            )
            self.curr_gripper_telemetry = getattr(
                self.robot_node,
                f"current_{self.robot_prefix}_gripper_telemetry",
                {},
            )
        self.curr_robot_timestamp = getattr(
            self.robot_node, f"current_{self.robot_prefix}_robot_timestamp"
        )
        getter = getattr(self.robot_node, 'get_gripper_observation', None)
        sample = getter(self.robot_prefix) if callable(getter) else None
        if not isinstance(sample, dict):
            sample = dict(opening_fraction=float(self.curr_gripper_pos), valid=True,
                          measured=None, observed_at=None, age_s=None, source='legacy')
        interface = getattr(self, 'gripper_interface', None)
        self.curr_gripper_observation = interface.observation(sample) if interface is not None else sample

    def _current_state_history_entry(self) -> np.ndarray:
        entry = np.empty(self.state_history_dim, dtype=np.float32)  
        entry[0] = self.curr_gripper_pos
        entry[1:4] = self.currforce
        entry[4:11] = self.currpos
        entry[11:14] = self.currtorque
        entry[14:20] = self.currvel
        return entry

    def _history_to_array(
        self, history, feature_dim: int, fallback_row: np.ndarray | None = None
    ) -> np.ndarray:
        if history is None or len(history) == 0:
            if fallback_row is None:
                return np.zeros((self.history_len, feature_dim), dtype=np.float32)
            return np.repeat(fallback_row[None], self.history_len, axis=0)

        stacked = np.asarray(list(history), dtype=np.float32)
        if stacked.ndim == 1:
            stacked = stacked.reshape(1, -1)

        if stacked.shape[0] < self.history_len:
            pad = np.repeat(stacked[:1], self.history_len - stacked.shape[0], axis=0)
            stacked = np.concatenate((pad, stacked), axis=0)

        return stacked[-self.history_len :]

    def _get_obs(self) -> dict:
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
            "wrench_history": self._history_to_array(self.wrench_history, 6),
            "state_history": self._history_to_array(
                self.state_history,
                self.state_history_dim,
                fallback_row=self._current_state_history_entry(),
            ),
        }
        return copy.deepcopy(dict(state=state_observation))
