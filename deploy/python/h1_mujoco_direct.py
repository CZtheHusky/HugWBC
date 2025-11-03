import time
import threading
from typing import Optional

import numpy as np
import mujoco
import mujoco.viewer

from consts import *
from utils import quat_rotate_inverse, transform_imu_data
import mujoco_config as mujoco_config


def cmd2out1step(target_cmd, tgt, dt, scale, decay_rate):
    step = scale * dt * np.exp(-np.abs(target_cmd) * decay_rate)
    if (tgt - target_cmd) <= -step:
        return tgt + step
    elif (tgt - target_cmd) >= step:
        return tgt - step
    else:
        return target_cmd


class H1MujocoDirect:
    def __init__(
        self,
        Control_Mode,
        interrupt_flag,
        keys_share,
        control_dt: float = 0.02,
        max_joint_velocity: float = 0.5,
        weight_rate: float = 0.2,
    ) -> None:
        # Shared control flags (kept for API parity)
        self.Control_Mode = Control_Mode
        self.interrupt_flag = interrupt_flag
        self.keys_share = keys_share

        # Load mujoco model
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_config.ROBOT_SCENE)
        self.mj_data = mujoco.MjData(self.mj_model)
        if mujoco_config.ROBOT == "h1" or mujoco_config.ROBOT == "g1":
            self._track_body_id = self.mj_model.body("torso_link").id
        else:
            self._track_body_id = self.mj_model.body("base_link").id

        # Viewer (passive)
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        # Timing
        self.control_dt = control_dt
        self.sim_dt = mujoco_config.SIMULATE_DT
        self.mj_model.opt.timestep = self.sim_dt
        self.sim_steps_per_control = max(1, int(round(self.control_dt / self.sim_dt)))

        # Gains per actuator (nu == number of motors)
        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = 3 * self.num_motor

        self.kp = np.zeros(self.num_motor, dtype=np.float32)
        self.kd = np.zeros(self.num_motor, dtype=np.float32)
        self._initialize_motor_gains()

        # State used by observation and control
        self.weight_rate = weight_rate
        self.max_joint_velocity = max_joint_velocity
        self.max_joint_delta = max_joint_velocity * control_dt
        self.delta_weight = weight_rate * control_dt
        self.gravity_vec = np.array([0, 0, -1])
        self.gait_index = 0

        self.default_pos = np.array([v for _, v in default_joint_angles.items()], dtype=np.float32)
        self.last_action = np.zeros(len(WholeBodyJoints), dtype=np.float32)
        self.commands = np.array([
            0, 0, 0,
            2.0, 0.5, 0.5,  # gait frequency, phase, duration
            0.15, 0.0,      # swing_height, body_height
            0.0, 0.0, 0.0   # body_pitch, waist_roll, interrupt flag (optional)
        ], dtype=np.float32)
        self.tgt_commands = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Mutex for stepping
        self._locker = threading.Lock()

        # For sim parity with existing flow in simulation, use predefined sequence
        if USE_SIM:
            self.Control_Mode.value = 4

        # Warm-up to initial posture
        self.set_to_target_pos(self.get_joint_pos())

    def _initialize_motor_gains(self) -> None:
        for name, idx in JointIndex.items():
            if name in ['kRightHipYaw', 'kRightHipRoll', 'kRightHipPitch', 'kLeftHipYaw', 'kLeftHipRoll', 'kLeftHipPitch']:
                self.kp[idx] = kp_hip
                self.kd[idx] = kd_hip
            elif name in ['kRightKnee', 'kLeftKnee', 'kWaistYaw']:
                self.kp[idx] = kp_knee_torso
                self.kd[idx] = kd_knee_torso
            elif name in ['kLeftAnkle', 'kRightAnkle']:
                self.kp[idx] = kp_ankle
                self.kd[idx] = kd_ankle
            elif name in ['kRightShoulderPitch', 'kRightShoulderRoll', 'kRightShoulderYaw', 'kRightElbow', 'kLeftShoulderPitch', 'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow']:
                self.kp[idx] = kp_arm
                self.kd[idx] = kd_arm
            else:
                self.kp[idx] = 0.0
                self.kd[idx] = 0.0

    def _apply_pd_control(self, q_des: np.ndarray, dq_des: Optional[np.ndarray] = None, tau_ff: Optional[np.ndarray] = None) -> None:
        if dq_des is None:
            dq_des = np.zeros_like(q_des)
        if tau_ff is None:
            tau_ff = np.zeros_like(q_des)

        # sensordata layout mirrors the bridge: [q (nu), dq (nu), tau (nu), imu...]
        q_meas = self.mj_data.sensordata[:self.num_motor]
        dq_meas = self.mj_data.sensordata[self.num_motor:self.num_motor * 2]

        ctrl = tau_ff + self.kp * (q_des - q_meas) + self.kd * (dq_des - dq_meas)
        self.mj_data.ctrl[:self.num_motor] = ctrl

    def _simulate_for_control_dt(self) -> None:
        for _ in range(self.sim_steps_per_control):
            with self._locker:
                mujoco.mj_step(self.mj_model, self.mj_data)
            if self.viewer.is_running():
                p = self.mj_data.xpos[self._track_body_id]
                self.viewer.cam.lookat[:] = p
                self.viewer.sync()

    def get_default_pos(self) -> np.ndarray:
        return self.default_pos

    def get_joint_pos(self) -> np.ndarray:
        q_meas = self.mj_data.sensordata[:self.num_motor]
        return np.array([q_meas[j] for j in WholeBodyJoints], dtype=np.float32)

    def set_to_target_pos(self, target_pos: Optional[np.ndarray] = None) -> None:
        current_jpos_des = self.get_joint_pos()
        if target_pos is None:
            target_pos = current_jpos_des
        for _ in range(500):
            for j in range(len(target_pos)):
                delta = np.clip(target_pos[j] - current_jpos_des[j], -self.max_joint_delta, self.max_joint_delta)
                current_jpos_des[j] += delta
            action = (current_jpos_des - self.default_pos) * (1.0 / ACTION_SCALE)
            self.step(action)
            time.sleep(0.005)

    def set_default_pos(self, target_pos: Optional[np.ndarray] = None) -> None:
        # API compatibility sugar
        self.set_to_target_pos(target_pos)

    def calculate_predefined_commands(self) -> None:
        # Mirror the bridge-driven behavior by cycling through predefined stages
        # Track time between calls
        if not hasattr(self, "_predef_idx"):
            self._predef_idx = 0
            self._predef_last_time = time.time()
        now = time.time()
        interval = Stage_Commands[self._predef_idx]["Interval"]
        if now - self._predef_last_time >= interval:
            self._predef_idx = (self._predef_idx + 1) % len(Stage_Commands)
            self._predef_last_time = now
        cmd_entry = Stage_Commands[self._predef_idx]
        self.commands[:11] = cmd_entry["Commands"]
        self.gait_index = cmd_entry.get("Gait_Index", self.gait_index)
        # Interrupt flag from commands (if present)
        if len(self.commands) > 10:
            self.interrupt_flag.value = 1 if self.commands[10] == 1 else 0

    def compute_obs(self) -> np.ndarray:
        if self.Control_Mode.value == 4:
            self.calculate_predefined_commands()
        # Smooth target commands
        for i in range(3):
            self.tgt_commands[i] = cmd2out1step(self.commands[i], self.tgt_commands[i], self.control_dt, 3.0, 0.6)

        q_meas = self.mj_data.sensordata[:self.num_motor]
        dq_meas = self.mj_data.sensordata[self.num_motor:self.num_motor * 2]

        dof_pos = (np.array([q_meas[j] for j in WholeBodyJoints], dtype=np.float32) - self.default_pos) * obs_scale_interrupt.dof_pos
        dof_vel = np.array([dq_meas[j] for j in WholeBodyJoints], dtype=np.float32) * obs_scale_interrupt.dof_vel
        if MASK_INT and self.interrupt_flag.value:
            dof_pos[-8:] = 0
            dof_vel[-8:] = 0

        # IMU from sensors (quat, gyro) located after motor sensors
        quat = self.mj_data.sensordata[self.dim_motor_sensor + 0:self.dim_motor_sensor + 4]
        ang_vel = self.mj_data.sensordata[self.dim_motor_sensor + 4:self.dim_motor_sensor + 7]

        waist_yaw = q_meas[JointIndex["kWaistYaw"]]
        waist_yaw_omega = dq_meas[JointIndex["kWaistYaw"]]
        gravity_vec, ang_vel_world = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        obs = np.concatenate(
            (
                ang_vel_world * obs_scale_interrupt.ang_vel,
                gravity_vec,
                dof_pos,
                dof_vel,
                self.last_action,
                self.tgt_commands * obs_scale_interrupt.commands[:3],
                self.commands[3:] * obs_scale_interrupt.commands[3:],
            ), axis=-1
        )
        return obs.astype(np.float32)

    def step(self, action: Optional[np.ndarray] = None, interrupt_action: Optional[np.ndarray] = None, dq: float = 0.0, tau_ff: float = 0.0) -> None:
        # For non-control mode steps, just advance sim
        if action is None:
            # keep previous ctrl
            self._simulate_for_control_dt()
            return

        self.last_action[:] = action

        # Desired targets per actuator
        q_des = np.zeros(self.num_motor, dtype=np.float32)
        dq_des = np.zeros(self.num_motor, dtype=np.float32)
        tau_ff_arr = np.zeros(self.num_motor, dtype=np.float32) + tau_ff

        # Map policy action -> WholeBody joints
        for j, motor_idx in enumerate(WholeBodyJoints[:11]):
            q_des[motor_idx] = action[j] * ACTION_SCALE + self.default_pos[j]
            dq_des[motor_idx] = dq

        # Arms (either interrupt_action or remainder of action)
        if interrupt_action is not None:
            for j, motor_idx in enumerate(WholeBodyJoints[11:]):
                q_des[motor_idx] = interrupt_action[j] * ACTION_SCALE + self.default_pos[11 + j]
                dq_des[motor_idx] = dq
            # Stronger gains for arms during interrupt (match original behavior)
            for name in ['kRightShoulderPitch', 'kRightShoulderRoll', 'kRightShoulderYaw', 'kRightElbow', 'kLeftShoulderPitch', 'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow']:
                idx = JointIndex[name]
                self.kp[idx] = 80
                self.kd[idx] = 1
        else:
            for j, motor_idx in enumerate(WholeBodyJoints[11:]):
                q_des[motor_idx] = action[11 + j] * ACTION_SCALE + self.default_pos[11 + j]
                dq_des[motor_idx] = dq
            # Reset gains to nominal arms values
            for name in ['kRightShoulderPitch', 'kRightShoulderRoll', 'kRightShoulderYaw', 'kRightElbow', 'kLeftShoulderPitch', 'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow']:
                idx = JointIndex[name]
                self.kp[idx] = kp_arm
                self.kd[idx] = kd_arm

        # Apply PD and step sim for control dt
        self._apply_pd_control(q_des=q_des, dq_des=dq_des, tau_ff=tau_ff_arr)
        self._simulate_for_control_dt()

    def get_gait_idx(self) -> int:
        return self.gait_index

    def finalize(self) -> None:
        # Bring gains to zero and stop
        self.kp[:] = 0.0
        self.kd[:] = 0.0
        # Keep one last step to settle
        self._apply_pd_control(q_des=np.zeros(self.num_motor, dtype=np.float32))
        self._simulate_for_control_dt()
        # viewer closes with process end; nothing explicit needed


