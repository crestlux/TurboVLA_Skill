"""Adapter between ``Rb3Env`` and the PLAIF partner controller.

The environment speaks absolute TCP targets in the cell base frame, with TCP
axes aligned to the controller's flange axes. Reset targets and feedback use
this native convention without a fixed rotation or a TCP-origin correction.
The partner controller's streaming lane consumes flange-frame delta pose
chunks, so absolute-to-delta conversion is still required at that boundary.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Callable

import numpy as np
import rclpy
from rb_env.plaif_messages import Move, Waypoint, Command, Sync
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, WrenchStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from .plaif_math import (
    absolute_targets_to_flange_deltas,
    normalized_pose,
    rotation_from_wxyz,
    rotation_to_wxyz,
)


SIDES = ("left", "right")
STATE_DISABLED = 1
STATE_ENABLED = 3
STATE_ON_TASK = 5
STATE_FAULTED = 7
TASK_IDLE = 4
TASK_FOLLOW = 10
TASK_SKU_FORCE_CONTROL = 18


def _stamp_seconds(msg) -> float:
    stamp = msg.header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 else time.time()


def flange_delta_pose(delta) -> Pose:
    """Encode a flange-frame translation/rotation-vector delta as a ROS Pose."""

    value = np.asarray(delta, dtype=float)
    if value.shape != (6,) or not np.all(np.isfinite(value)):
        raise ValueError("follow delta must contain 6 finite values")
    quaternion = rotation_to_wxyz(Rotation.from_rotvec(value[3:]))
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = map(float, value[:3])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = map(
        float, quaternion
    )
    return msg


class PlaifNode(Node):
    """``Rb3Env``-compatible arm backend for ``plaif-partner-release``."""

    ROBOT_OBSERVATION_HZ = 500.0

    # Clip against the same live pose used for the final wire delta.
    handles_delta_clipping = True
    supports_action_chunks = True

    def __init__(
        self,
        platform: str = "monkey",
        task: str = "sku",
        wait_for_data: bool = True,
        state_timeout: float = 10.0,
        velocity_filter_alpha: float = 0.25,
        follow_period: float = 0.1,
        active_sides: tuple[str, ...] = SIDES,
        control_hz: float | None = None,
    ):
        active_sides = tuple(active_sides)
        if (not active_sides or len(set(active_sides)) != len(active_sides)
                or any(side not in SIDES for side in active_sides)):
            raise ValueError("active_sides must contain left and/or right without duplicates")
        if task != "sku":
            raise ValueError("PlaifNode currently requires task='sku'")
        if not 0.0 < velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in (0, 1]")
        if not np.isfinite(follow_period) or follow_period <= 0.0:
            raise ValueError("follow_period must be finite and positive")
        control_hz = 1.0 / follow_period if control_hz is None else float(control_hz)
        if not np.isfinite(control_hz) or control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive")

        super().__init__("plaif_env_backend")
        self.platform = platform.strip("/")
        if not self.platform:
            raise ValueError("platform must not be empty")
        self.task = task
        self.active_sides = active_sides
        self.wrench_observation_frame = "base_at_tcp"
        # Keep one nominal control period of 500 Hz robot observations.
        self.history_len = max(1, int(np.ceil(self.ROBOT_OBSERVATION_HZ / control_hz)))
        self.state_history_dim = 20
        self._velocity_alpha = float(velocity_filter_alpha)
        self._follow_period = float(follow_period)
        self._state_lock = threading.RLock()
        self._state_changed = threading.Condition(self._state_lock)
        self._lifecycle_lock = threading.RLock()
        self._last_follow_publish = {side: None for side in SIDES}
        self._last_follow_length = {side: 1 for side in SIDES}
        self._warned_target_wrench = False
        self._destroying = False
        self._task_machine_used = False
        self.callback_group = ReentrantCallbackGroup()

        self._init_state()
        self._init_ros_interfaces()

        self._executor = MultiThreadedExecutor(num_threads=6)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="plaif-node-spin",
            daemon=True,
        )
        self._spin_thread.start()

        if wait_for_data:
            try:
                self._wait_for_data(float(state_timeout))
            except Exception:
                self.destroy_node()
                raise
        self.get_logger().info(
            f"PlaifNode ready for /{self.platform}; follow task={self.task!r}."
        )

    def _init_state(self) -> None:
        for side in SIDES:
            setattr(self, f"current_{side}_robot_timestamp", None)
            setattr(self, f"current_{side}_qpos", None)
            setattr(self, f"current_{side}_qvel", None)
            setattr(self, f"current_{side}_qeff", None)
            setattr(self, f"current_{side}_tcp_pose", None)
            setattr(self, f"current_{side}_tcp_twist", np.zeros(6, dtype=float))
            setattr(self, f"current_{side}_wrench", None)
            setattr(self, f"current_{side}_force", None)
            setattr(self, f"current_{side}_torque", None)
            setattr(self, f"current_{side}_org_wrench", None)
            setattr(self, f"current_{side}_org_force", None)
            setattr(self, f"current_{side}_org_torque", None)
            setattr(self, f"current_{side}_jacobian", np.zeros((6, 6), dtype=float))
            setattr(self, f"current_{side}_dh_grip_length", 100.0)
            setattr(self, f"current_{side}_gripper_telemetry", {})
            setattr(self, f"history_{side}_state", deque(maxlen=self.history_len))
            setattr(self, f"history_{side}_wrench", deque(maxlen=self.history_len))
            setattr(self, f"diagnostic_{side}", {})
            setattr(self, f"_previous_{side}_pose_sample", None)
            setattr(self, f"_previous_{side}_joint_sample", None)

    def _state_qos(self) -> QoSProfile:
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.history_len,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

    @staticmethod
    def _command_qos() -> QoSProfile:
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

    def _init_ros_interfaces(self) -> None:
        state_qos = self._state_qos()
        command_qos = self._command_qos()
        self.follow_publishers = {}
        self.console_clients = {}
        self.move_clients = {}

        for side in self.active_sides:
            base = f"/{self.platform}/{side}"
            self.create_subscription(
                JointState,
                base + "/act/joint",
                lambda msg, s=side: self._joint_callback(s, msg),
                state_qos,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                PoseStamped,
                base + "/act/pose",
                lambda msg, s=side: self._pose_callback(s, msg),
                state_qos,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                WrenchStamped,
                base + "/act/wrench_tcp",
                lambda msg, s=side: self._wrench_callback(s, msg),
                state_qos,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                WrenchStamped,
                base + "/act/wrench_raw",
                lambda msg, s=side: self._raw_wrench_callback(s, msg),
                state_qos,
                callback_group=self.callback_group,
            )
            self.create_subscription(
                DiagnosticArray,
                base + "/state/diagnostic",
                lambda msg, s=side: self._diagnostic_callback(s, msg),
                state_qos,
                callback_group=self.callback_group,
            )
            self.follow_publishers[side] = self.create_publisher(
                PoseArray, base + "/cmd/follow", command_qos
            )
            self.console_clients[side] = self.create_client(
                Command, base + "/cmd/console", callback_group=self.callback_group
            )
            self.move_clients[side] = ActionClient(
                self, Move, base + "/cmd/move", callback_group=self.callback_group
            )

        cell = f"/{self.platform}/cell/cmd"
        self.cell_console_client = self.create_client(
            Command, cell + "/console", callback_group=self.callback_group
        )
        self.sync_client = self.create_client(
            Sync, cell + "/sync", callback_group=self.callback_group
        )

    def _joint_callback(self, side: str, msg: JointState) -> None:
        if len(msg.position) < 6:
            self.get_logger().warning(f"Ignoring short {side} JointState")
            return
        stamp = _stamp_seconds(msg)
        qpos = np.asarray(msg.position[:6], dtype=float)
        effort = np.zeros(6, dtype=float)
        if len(msg.effort) >= 6:
            effort[:] = msg.effort[:6]

        with self._state_changed:
            previous = getattr(self, f"_previous_{side}_joint_sample")
            qvel = np.zeros(6, dtype=float)
            if previous is not None:
                previous_stamp, previous_qpos, previous_qvel = previous
                dt = stamp - previous_stamp
                if 1e-4 <= dt <= 0.25:
                    estimate = (qpos - previous_qpos) / dt
                    a = self._velocity_alpha
                    qvel = a * estimate + (1.0 - a) * previous_qvel
            setattr(self, f"current_{side}_qpos", qpos)
            setattr(self, f"current_{side}_qvel", qvel)
            setattr(self, f"current_{side}_qeff", effort)
            setattr(self, f"current_{side}_robot_timestamp", stamp)
            setattr(self, f"_previous_{side}_joint_sample", (stamp, qpos.copy(), qvel.copy()))
            self._state_changed.notify_all()

    def _pose_callback(self, side: str, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        pose = normalized_pose([p.x, p.y, p.z, q.w, q.x, q.y, q.z])
        stamp = _stamp_seconds(msg)

        with self._state_changed:
            previous = getattr(self, f"_previous_{side}_pose_sample")
            twist = getattr(self, f"current_{side}_tcp_twist").copy()
            if previous is not None:
                previous_stamp, previous_pose = previous
                dt = stamp - previous_stamp
                if 1e-4 <= dt <= 0.25:
                    linear = (pose[:3] - previous_pose[:3]) / dt
                    current_r = rotation_from_wxyz(pose[3:])
                    previous_r = rotation_from_wxyz(previous_pose[3:])
                    angular = (current_r * previous_r.inv()).as_rotvec() / dt
                    estimate = np.concatenate((linear, angular))
                    a = self._velocity_alpha
                    twist = a * estimate + (1.0 - a) * twist
            setattr(self, f"current_{side}_tcp_pose", pose)
            setattr(self, f"current_{side}_tcp_twist", twist)
            setattr(self, f"current_{side}_robot_timestamp", stamp)
            setattr(self, f"_previous_{side}_pose_sample", (stamp, pose.copy()))
            self._state_changed.notify_all()

    def _wrench_callback(self, side: str, msg: WrenchStamped) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        wrench_tool = np.asarray([f.x, f.y, f.z, t.x, t.y, t.z], dtype=float)
        with self._state_changed:
            pose = getattr(self, f"current_{side}_tcp_pose")
            if pose is None:
                return
            base_from_tool = rotation_from_wxyz(pose[3:])
            wrench_base = np.concatenate(
                (base_from_tool.apply(wrench_tool[:3]), base_from_tool.apply(wrench_tool[3:]))
            )
            setattr(self, f"current_{side}_wrench", wrench_base)
            setattr(self, f"current_{side}_force", wrench_base[:3])
            setattr(self, f"current_{side}_torque", wrench_base[3:])
            getattr(self, f"history_{side}_wrench").append(wrench_base.copy())
            # One state snapshot per FT sample; joint/pose callbacks only update
            # the latest values, so they cannot evict FT-aligned history rows.
            self._append_state_history_locked(side)
            self._state_changed.notify_all()

    def _raw_wrench_callback(self, side: str, msg: WrenchStamped) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        wrench = np.asarray([f.x, f.y, f.z, t.x, t.y, t.z], dtype=float)
        with self._state_changed:
            setattr(self, f"current_{side}_org_wrench", wrench)
            setattr(self, f"current_{side}_org_force", wrench[:3])
            setattr(self, f"current_{side}_org_torque", wrench[3:])

    def _diagnostic_callback(self, side: str, msg: DiagnosticArray) -> None:
        for status in msg.status:
            values = {item.key: item.value for item in status.values}
            if "state" not in values:
                continue
            with self._state_changed:
                setattr(self, f"diagnostic_{side}", values)
                self._state_changed.notify_all()
            return

    def _append_state_history_locked(self, side: str) -> None:
        force = getattr(self, f"current_{side}_force")
        torque = getattr(self, f"current_{side}_torque")
        pose = getattr(self, f"current_{side}_tcp_pose")
        twist = getattr(self, f"current_{side}_tcp_twist")
        if force is None or torque is None or pose is None:
            return
        state = np.empty(self.state_history_dim, dtype=np.float32)
        state[0] = getattr(self, f"current_{side}_dh_grip_length") / 100.0
        state[1:4] = force
        state[4:11] = pose
        state[11:14] = torque
        state[14:20] = twist
        getattr(self, f"history_{side}_state").append(state)

    def _missing_data(self) -> list[str]:
        missing = []
        for side in self.active_sides:
            for name in ("qpos", "tcp_pose", "force"):
                if getattr(self, f"current_{side}_{name}") is None:
                    missing.append(f"{side}.{name}")
            if not getattr(self, f"diagnostic_{side}"):
                missing.append(f"{side}.diagnostic")
        return missing

    def _wait_for_data(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._state_changed:
            while True:
                missing = self._missing_data()
                if not missing:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "PLAIF controller state timeout; missing " + ", ".join(missing)
                    )
                self._state_changed.wait(min(0.1, remaining))

    @staticmethod
    def _int_value(values: dict, key: str, default: int = -1) -> int:
        try:
            return int(values.get(key, default))
        except (TypeError, ValueError):
            return default

    def _state_task(self, side: str) -> tuple[int, int]:
        with self._state_lock:
            diag = dict(getattr(self, f"diagnostic_{side}"))
        return self._int_value(diag, "state"), self._int_value(diag, "task")

    def _wait_until(
        self, side: str, predicate: Callable[[int, int], bool], timeout: float
    ) -> tuple[int, int]:
        deadline = time.monotonic() + timeout
        with self._state_changed:
            while True:
                state, task = self._state_task(side)
                if predicate(state, task):
                    return state, task
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"{side} state transition timeout (state={state}, task={task})"
                    )
                self._state_changed.wait(min(0.05, remaining))

    def _wait_future(self, future, timeout: float, what: str):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {what}")
            time.sleep(0.002)
        result = future.result()
        if result is None:
            raise RuntimeError(f"{what} returned no result")
        return result

    def _call_console(self, command: str, side: str | None = None, timeout: float = 50.0):
        client = self.cell_console_client if side is None else self.console_clients[side]
        if not client.wait_for_service(timeout_sec=5.0):
            target = "cell" if side is None else side
            raise RuntimeError(f"{target} console service is unavailable")
        response = self._wait_future(
            client.call_async(Command.Request(command=command)),
            timeout,
            f"console command {command!r}",
        )
        if not response.accepted:
            raise RuntimeError(f"console command {command!r} refused: {response.message}")
        return response

    def _ensure_task_machine(self, side: str) -> None:
        with self._lifecycle_lock:
            if self._destroying:
                raise RuntimeError("PlaifNode is shutting down")
            state, _ = self._state_task(side)
            if state == STATE_FAULTED:
                raise RuntimeError(f"{side} controller is Faulted")
            if state == STATE_DISABLED:
                self._call_console("enable")
                state, _ = self._wait_until(side, lambda s, _t: s == STATE_ENABLED, 50.0)
            elif state not in (STATE_ENABLED, STATE_ON_TASK):
                state, _ = self._wait_until(
                    side,
                    lambda s, _t: s in (STATE_DISABLED, STATE_ENABLED, STATE_ON_TASK, STATE_FAULTED),
                    50.0,
                )
                if state == STATE_DISABLED:
                    self._call_console("enable")
                    self._wait_until(side, lambda s, _t: s == STATE_ENABLED, 50.0)
                elif state == STATE_FAULTED:
                    raise RuntimeError(f"{side} controller is Faulted")
            state, _ = self._state_task(side)
            if state == STATE_ENABLED:
                # Keep shutdown responsible even if task-on times out or is
                # interrupted after the controller has accepted the command.
                self._task_machine_used = True
                self._call_console("task on")
                self._wait_until(side, lambda s, _t: s == STATE_ON_TASK, 10.0)
            elif state == STATE_ON_TASK:
                self._task_machine_used = True

    def _ensure_sku(self, side: str) -> None:
        self._ensure_task_machine(side)
        state, task = self._state_task(side)
        if state != STATE_ON_TASK:
            raise RuntimeError(f"{side} is not in the task machine")
        if task == TASK_SKU_FORCE_CONTROL:
            return
        if task != TASK_IDLE:
            self._call_console("task clear", side=side, timeout=10.0)
            self._wait_until(
                side, lambda s, t: s == STATE_ON_TASK and t == TASK_IDLE, 10.0
            )
        self._call_console("task sku", side=side, timeout=10.0)
        self._wait_until(
            side,
            lambda s, t: s == STATE_ON_TASK and t == TASK_SKU_FORCE_CONTROL,
            10.0,
        )

    def _assert_healthy(self, side: str) -> None:
        with self._state_lock:
            diag = dict(getattr(self, f"diagnostic_{side}"))
        if not diag:
            raise RuntimeError(f"{side} diagnostic state is unavailable")
        faults = []
        for key in ("last_error", "estopped", "self_collision", "ext_collision"):
            if self._int_value(diag, key, 0) != 0:
                faults.append(f"{key}={diag.get(key)}")
        if self._int_value(diag, "device_error", 0) != 0:
            faults.append(f"device_error={diag.get('device_error')}")
        if self._int_value(diag, "servo_on", 0) != 1:
            faults.append(f"servo_on={diag.get('servo_on')}")
        if faults:
            raise RuntimeError(f"{side} controller is not healthy: " + ", ".join(faults))

    # Compatibility with the former ros2_control backend.  This enters the
    # streaming machine but deliberately stays in Idle; the first follow
    # command explicitly selects SkuForceControl.
    def turn_cartesian_controller(self, side: str) -> None:
        self._validate_side(side)
        self._ensure_task_machine(side)

    def turn_cartesian_motion_controller(self, side: str) -> None:
        self.turn_cartesian_controller(side)

    def turn_off_controllers(self, side: str) -> None:
        self._validate_side(side)
        state, _ = self._state_task(side)
        if state == STATE_ON_TASK:
            # Leaving the task machine is a cell command in the controller API.
            self._call_console("task stop", timeout=10.0)

    def turn_position_controller(self, side: str) -> None:
        self.turn_off_controllers(side)

    def call_zero_ft(self) -> None:
        # The partner controller tares on entry to the task machine.
        self._call_console("task off")
        self._call_console("task on")

    def publish_compliance_target_frame(
        self,
        target_frame,
        target_wrench,
        side: str,
        is_clip: bool = True,
        target_duration: float = 0.0,
        start_frame=None,
        max_translation_step=None,
        max_rotation_step=None,
    ) -> dict:
        """Send one absolute pose [7] or a short pose chunk [H, 7].

        The first delta uses measured TCP feedback; subsequent deltas use the
        bounded preceding commands. start_frame remains ignored.
        """
        del is_clip, target_duration, start_frame
        self._validate_side(side)
        for limit in (max_translation_step, max_rotation_step):
            if limit is not None and (not np.isfinite(limit) or limit < 0):
                raise ValueError("step limits must be finite and non-negative")
        targets = np.asarray(target_frame, dtype=float)
        if targets.ndim == 1:
            targets = normalized_pose(targets)[None, :]
        elif (targets.ndim == 2 and targets.shape[1] == 7
              and 1 <= len(targets) <= 50):
            targets = np.stack([normalized_pose(target) for target in targets])
        else:
            raise ValueError("target_frame must be [7] or [H, 7] with 1 <= H <= 50")
        wrench = np.asarray(target_wrench, dtype=float)
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise ValueError("target_wrench must contain 6 finite values")
        if np.any(np.abs(wrench) > 1e-12) and not self._warned_target_wrench:
            self.get_logger().warning(
                "PlaifNode ignores target_wrench; SkuForceControl owns its force law."
            )
            self._warned_target_wrench = True

        # A long local gap means Sku may have parked to Idle.  Give the 50 Hz
        # diagnostic stream one sample to expose that transition before
        # selecting the task, preventing automatic entry into factory Follow.
        previous_publish = self._last_follow_publish[side]
        previous_length = getattr(self, "_last_follow_length", {}).get(side, 1)
        gap_periods = previous_length + 2 if previous_length > 1 else 2
        if previous_publish is not None and time.monotonic() - previous_publish > gap_periods * self._follow_period:
            try:
                self._wait_until(side, lambda _s, t: t != TASK_SKU_FORCE_CONTROL, 0.06)
            except TimeoutError:
                pass
        self._ensure_sku(side)
        self._assert_healthy(side)

        with self._state_lock:
            current = getattr(self, f"current_{side}_tcp_pose")
            if current is None:
                raise RuntimeError(f"current {side} TCP pose is unavailable")
            current = current.copy()

        deltas = absolute_targets_to_flange_deltas(
            current, targets,
            max_translation_step=max_translation_step,
            max_rotation_step=max_rotation_step,
        )
        message = PoseArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "flange"
        message.poses = [flange_delta_pose(delta) for delta in deltas]
        self.follow_publishers[side].publish(message)
        self._last_follow_publish[side] = time.monotonic()
        if not hasattr(self, "_last_follow_length"):
            self._last_follow_length = {}
        self._last_follow_length[side] = len(targets)
        # Record the bounded commands using the exact feedback snapshot used
        # on the wire, not a second estimate after feedback has moved.
        pose = current.copy()
        sent_targets = []
        for delta in deltas:
            rotation = rotation_from_wxyz(pose[3:])
            pose = np.concatenate((
                pose[:3] + rotation.apply(delta[:3]),
                rotation_to_wxyz(rotation * Rotation.from_rotvec(delta[3:])),
            ))
            sent_targets.append(pose)
        return {"start_pose": current, "requested_poses": targets.copy(),
                "target_poses": np.stack(sent_targets),
                "flange_deltas": deltas.copy()}


    def clear_follow_chunk(self, side: str) -> None:
        """Discard a multi-pose tail at an episode/reset boundary."""
        if self._last_follow_length.get(side, 1) <= 1:
            return
        state, task = self._state_task(side)
        if state == STATE_ON_TASK and task in (TASK_SKU_FORCE_CONTROL, TASK_FOLLOW):
            self._call_console("task clear", side=side, timeout=10.0)
            self._wait_until(side, lambda s, t: s == STATE_ON_TASK and t == TASK_IDLE, 10.0)
        self._last_follow_length[side] = 1
        self._last_follow_publish[side] = None

    def move_linear(
        self,
        target_pose,
        side: str,
        speed_mps: float = 0.05,
        timeout: float = 30.0,
    ) -> None:
        """Run one absolute base-frame MOVL and wait for its action result."""

        self._validate_side(side)
        target = normalized_pose(target_pose)
        if not np.isfinite(speed_mps) or speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive")
        self._ensure_task_machine(side)

        # A follow lane naturally parks after two input periods.  Wait for that
        # cheap transition before clearing this arm's unit explicitly.
        try:
            self._wait_until(
                side,
                lambda s, t: s == STATE_ON_TASK and t == TASK_IDLE,
                max(0.35, 3.0 * self._follow_period),
            )
        except TimeoutError:
            state, _ = self._state_task(side)
            if state == STATE_ON_TASK:
                self._call_console("task clear", side=side, timeout=10.0)
            self._ensure_task_machine(side)
            self._wait_until(
                side,
                lambda s, t: s == STATE_ON_TASK and t == TASK_IDLE,
                10.0,
            )

        client = self.move_clients[side]
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"{side} Move action server is unavailable")
        goal = Move.Goal()
        goal.kind = Move.Goal.MOVL
        waypoint = Waypoint()
        waypoint.v = list(target[:3]) + list(
            rotation_from_wxyz(target[3:]).as_euler("xyz")
        )
        goal.waypoints = [waypoint]
        goal.velocity_pct = [float(speed_mps)]
        goal.velocity_is_abs = [True]

        goal_handle = self._wait_future(
            client.send_goal_async(goal), 5.0, f"register {side} MOVL"
        )
        if not goal_handle.accepted:
            raise RuntimeError(f"{side} MOVL was rejected at registration")
        # Acceptance is not proof that asynchronous pre-planning succeeded.
        # Keep its result from the start so a planner rejection is not hidden
        # by the subsequent Sync error ("no goal is registered").
        result_future = goal_handle.get_result_async()
        if result_future.done():
            self._check_move_result(side, result_future.result(), "before release")
            return

        if not self.sync_client.wait_for_service(timeout_sec=5.0):
            goal_handle.cancel_goal_async()
            raise RuntimeError("cell Sync service is unavailable")
        target_side = Sync.Request.LEFT if side == "left" else Sync.Request.RIGHT
        sync = self._wait_future(
            self.sync_client.call_async(Sync.Request(target=target_side)),
            5.0,
            f"release {side} MOVL",
        )
        if not sync.accepted:
            try:
                early_result = self._wait_future(
                    result_future, 0.5, f"{side} MOVL planning result"
                )
            except TimeoutError:
                early_result = None
            if early_result is not None:
                self._check_move_result(side, early_result, "during registration/planning")
            goal_handle.cancel_goal_async()
            raise RuntimeError(f"{side} MOVL release refused: {sync.message}")

        try:
            wrapped_result = self._wait_future(
                result_future, timeout, f"complete {side} MOVL"
            )
        except TimeoutError:
            goal_handle.cancel_goal_async()
            raise
        self._check_move_result(side, wrapped_result, "during execution")

    @staticmethod
    def _check_move_result(side: str, wrapped_result, phase: str) -> None:
        if wrapped_result is None:
            raise RuntimeError(f"{side} MOVL returned no result {phase}")
        if wrapped_result.result.result != Move.Result.DONE:
            code = wrapped_result.result.result
            names = {getattr(Move.Result, name): name for name in
                     ("DONE", "REJECTED", "ABORTED", "TIMEOUT", "SINGULARITY")}
            raise RuntimeError(
                f"{side} MOVL failed {phase}: result={names.get(code, code)}, "
                f"completed={wrapped_result.result.completed}. "
                "Check the controller planning log for the IK/path rejection reason."
            )

    def _validate_side(self, side: str) -> None:
        if side not in self.active_sides:
            raise ValueError(f"Inactive or invalid side: {side!r}")

    def destroy_node(self, *args, **kwargs):
        with self._lifecycle_lock:
            if self._destroying:
                return True
            self._destroying = True
            try:
                if self._task_machine_used:
                    # Console responses and state feedback still need the
                    # executor. Task-off leaves the cell Enabled (servo on).
                    self._call_console("task off", timeout=10.0)
                    for side in self.active_sides:
                        self._wait_until(
                            side,
                            lambda state, _task: state in (STATE_ENABLED, STATE_DISABLED),
                            10.0,
                        )
            except Exception as exc:
                self.get_logger().error(
                    f"PLAIF task off failed during shutdown; task mode may still be active: {exc}"
                )
            finally:
                try:
                    self._executor.remove_node(self)
                    self._executor.shutdown(timeout_sec=5.0)
                    if self._spin_thread.is_alive():
                        self._spin_thread.join(timeout=5.0)
                finally:
                    destroyed = super().destroy_node(*args, **kwargs)
            return destroyed
