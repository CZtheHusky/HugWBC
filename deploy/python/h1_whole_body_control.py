import time
import numpy as np
import math
import struct
import os

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_._LowCmd_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_._LowState_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from consts import *
from utils import *
def cmd2out1step(target_cmd, tgt, dt, scale, decay_rate):
    step = scale * dt * np.exp(-np.abs(target_cmd) * decay_rate)
    if (tgt - target_cmd <= -step):
        return tgt + step
    elif (tgt - target_cmd >= step):
        return tgt -step
    else:
        return target_cmd


class H1:
    def __init__(
        self,
        Control_Mode,
        interrupt_flag,
        keys_share,
        control_dt=0.02,
        max_joint_velocity=0.5,
        weight_rate=0.2,
    ) -> None:
        if USE_SIM:
            import mujoco_config
            ChannelFactoryInitialize(
               id=mujoco_config.DOMAIN_ID, networkInterface=mujoco_config.INTERFACE
            )
        else:
            ChannelFactoryInitialize(
                networkInterface=net_interface
            )  # TODO: check on your machine
            print("REAL CODE")

        # Create a publisher to publish the data defined in UserData class
        # self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        # Create a subscripber to obtain low state
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(None, 10)

        print("FINISH SUB AND PUB")
        # Create a message
        self.msg = unitree_go_msg_dds__LowCmd_()
        self.msg.head[0] = 0xFE
        self.msg.head[1] = 0xEF
        self.msg.level_flag = 0xFF
        self.msg.gpio = 0
        self.set_motor_gains()

        self.crc = CRC()
        # self.commands = np.array([0,0,0], dtype=np.float32)
        self.tgt_commands = np.array([0,0,0], dtype=np.float32)
        
        self.commands = np.array([0, 0, 0, 
                                  1.5, 0.5, 0.5,  # Frequency, phases, duration
                                  0.15, 0.0,      # swing_height body_height
                                  0.0, 0.0], dtype=np.float32) # pitch waist
        self.clock_input = np.array([0, 0], dtype=np.float32)
        self.low_speed_level = np.array([1.0, 0.6, 0.6], dtype=np.float32)
        self.high_speed_level = np.array([2.0, 0.6, 1.0], dtype=np.float32)
        self.commands_level = self.low_speed_level.copy()
        self.velocity_level = 0.0
        self.gait_index = 0  # 0 standing 1 walking 2 hopping
        self.gait_param_index = 99
        self.global_phase = 0.0
        self.gait_frequecy = 1.0
        self.update_count = 0
        self.joystick_buffer = []


        #Joystick 
        self.key_state = key_state.copy()
        self.stick_state = {
            'lx': 0,
            'ly': 0,
            'rx': 0,
            'ry': 0
        }

        self.last_key_state = self.key_state.copy()
        self.last_stick_state = self.stick_state.copy()
        self.keys_share = keys_share

        self.last_action = np.zeros(19)
        self.jump_air_leg = 0 # 0: left leg in the air for jumping

        self.Control_Mode = Control_Mode # 0: Initialize Mode; 1: Default Pos Mode; 2: Controlling by policy; 3: Finalize Mode
        self.interrupt_flag = interrupt_flag

        self.init_joint_pos = self.get_joint_pos()  # 19
        self.final_joint_pos = self.init_joint_pos.copy()

        self.control_dt = control_dt
        self.max_joint_velocity = max_joint_velocity
        self.weight_rate = weight_rate
        self.max_joint_delta = max_joint_velocity * control_dt
        print("max_joint_delta: ", self.max_joint_delta)
        self.delta_weight = weight_rate * control_dt
        self.gravity_vec = np.array([0,0,-1])
        self.default_pos = np.array(
            [v for k,v in default_joint_angles.items()]
        ) 
        print("FINISH SETUP")
        self.init()

        self.predefined_command_stage = -1
        self.predefined_command_current_stage_start_time = -1

    #Key State:        
    def _on_press(self, key):
        return self.key_state[key] and (not self.last_key_state[key])
    def _on_release(self, key):
        return (not self.key_state[key]) and self.last_key_state[key]
    def _pressed(self, key):
        return self.key_state[key] and self.last_key_state[key]
    def _released(self, key):
        return (not self.key_state[key]) and (not self.last_key_state[key])

 
    def WirelessControlFrom_msg(self, msg): # msg : types.array[types.uint8, 40]
        
        unpacked_data = struct.unpack('<2B H 5f 16B', msg)
        self.keys_share[:] = msg[:]
        self.last_key_state = self.key_state.copy()
        self.last_stick_state = self.stick_state.copy()

        self.stick_state['lx'] = unpacked_data[3]
        self.stick_state['ly'] = unpacked_data[7]
        self.stick_state['rx'] = unpacked_data[4]
        self.stick_state['ry'] = unpacked_data[5]
        
        value = unpacked_data[2]
        for i,(k,v) in enumerate(self.key_state.items()):
            self.key_state[k] = (value & (1<<i)) >> i
        
        self.mode_state_machine()

    def mode_state_machine(self):
        # Change Control Mode
        if self.Control_Mode.value == 0:
            # In zero_torques_mode
            if self.key_state["L1"] and self.key_state["R2"] and (self._on_press("L1") or self._on_press("R2")):
                self.Control_Mode.value = 1
            if self.key_state["L2"] and self.key_state["R2"]:
                self.Control_Mode.value = 3
        elif self.Control_Mode.value == 1:
            # In Default Pos
            if self.key_state["R1"] and self.key_state["L1"] and (self._on_press("L1") or self._on_press("R1")):
                self.Control_Mode.value = 2
            if self.key_state["L2"] and self.key_state["R2"]:
                self.Control_Mode.value = 3
        elif self.Control_Mode.value == 2:
            # During Controlling
            if self.key_state["L1"] and self.key_state["R2"] and (self._on_press("L1") or self._on_press("R2")):
                self.Control_Mode.value = 1
            if self.key_state["R2"] and self.key_state["L2"]:
                self.Control_Mode.value = 3
            if self.key_state["L2"] and self.key_state["R1"] and (self._on_press("L2") or self._on_press("R1")):
                self.Control_Mode.value = 4
                self.predefined_command_current_stage_start_time = time.time()
                self.predefined_command_stage = 0
                print("TO CONTROL MODE 4")
        elif self.Control_Mode.value == 4: # Execute Predefined Control Sequence. Can only reach when start control.
            if self.key_state["L1"] and self.key_state["R2"] and (self._on_press("L1") or self._on_press("R2")):
                self.Control_Mode.value = 1 # Back to Default.
            if self.key_state["R1"] and self.key_state["L1"] and (self._on_press("L1") or self._on_press("R1")):
                self.Control_Mode.value = 2 # Back to Control
            if self.key_state["R2"] and self.key_state["L2"]:
                self.Control_Mode.value = 3 # Terminal
        if USE_SIM:
            self.Control_Mode.value = 4
       # print(self.Control_Mode.value)

    def calculate_predefined_commands(self):
        current_time = time.time()
        # print(current_time, self.predefined_command_stage)
        if current_time - self.predefined_command_current_stage_start_time > Stage_Commands[self.predefined_command_stage]["Interval"] and self.predefined_command_stage < len(Stage_Commands)-1:
            self.predefined_command_stage += 1
            self.predefined_command_current_stage_start_time = current_time
        if USE_SIM and self.predefined_command_stage == len(Stage_Commands)-1:
            self.predefined_command_stage = 0
            self.predefined_command_current_stage_start_time = current_time

        self.commands[:11] = Stage_Commands[self.predefined_command_stage]["Commands"]
        self.gait_index = Stage_Commands[self.predefined_command_stage]["Gait_Index"]
        if self.commands[10] == 1:
            self.interrupt_flag.value = 1
        if self.commands[10] == 0:
            self.interrupt_flag.value = 0


        # print(self.commands, self.gait_index)
        # if len(self.commands == 11): self.commands[-1] = 0

    def state_machine(self):
         
        self.commands[0] = self.stick_state['ly'] * self.commands_level[0]
        self.commands[0] = max(-0.6, self.commands[0])
        self.commands[1] = -self.stick_state['lx'] * self.commands_level[1]
        self.commands[2] = -self.stick_state['rx'] * self.commands_level[2]
        #print("self.stick_state:", self.stick_state)
        # Gait
        if self._pressed('L1'):
            if self._on_press('A'): # Walking
                self.gait_index = 1
                self.commands[4] = 0.5
                print("Walking")
            elif self._on_press('B'):
                self.gait_index = 2
                self.commands[4] = 0 # Hopping
                print("Hopping")
            elif self._on_press('X'): # Jumping
                self.commands[4] = 0
                if self.jump_air_leg == 0:
                    self.gait_index = 3
                    print("Jumping left")
                else:
                    self.gait_index = 4
                    print("Jumping right")
            elif self._on_press('Y'): # Standing
                self.gait_index = 0
                self.commands[4] = 0
                print("Standing")

        # Body Control:
        if self._pressed("L2"):
            if self._on_press('up'):
                self.commands[7] = min(self.commands[7]+0.05, commands_ranges['body_height'][1]) 
                print("Increase Body Height 0.05m")
            elif self._on_press("down"):
                self.commands[7] = max(self.commands[7]-0.05, commands_ranges['body_height'][0])
                print("Decrease Body Height 0.05m")
            elif self._on_press("left"):
                self.commands[9] = min(self.commands[9]+0.1, commands_ranges['waist_roll'][1])
                print("Turning left waist 0.1 rad")
            elif self._on_press("right"):
                self.commands[9] = max(self.commands[9]-0.1, commands_ranges['waist_roll'][0])
                print("Turning right waist 0.1 rad")
            elif self._on_press("Y"):
                self.commands[8] = max(self.commands[8]-0.1, commands_ranges['body_pitch'][0])
                print("Turning up pitch 0.1 rad")
            elif self._on_press("A"):
                self.commands[8] = min(self.commands[8]+0.1, commands_ranges['body_pitch'][1])
                print("Turning down pitch 0.1 rad")

        # Foot Control
        if self._pressed("R1"):
            if self._on_press("Y"):
                self.commands[6] = min(self.commands[6]+0.05, commands_ranges['foot_swing_height'][1])
                print("Raise foot swing height 0.05m")
            elif self._on_press("A"):
                self.commands[6] = max(self.commands[6]-0.05, commands_ranges['foot_swing_height'][0])
                print("Put down foot swing height 0.05m")
            elif self._on_press("up"):
                self.commands[3] = min(self.commands[3]+0.25, commands_ranges['gait_frequency'][1])
                print("Increase gait frequency 0.25 Hz")
            elif self._on_press("down"):
                self.commands[3] = max(self.commands[3]-0.25, commands_ranges['gait_frequency'][0])
                print("Decrease gait frequency 0.25 Hz")
        
        # Switching Jumping / Interruption
        if self._pressed("R2"):
            if self.gait_index==4 and self._on_press("left"):
                self.jump_air_leg = 0
                self.gait_index = 3
                print("Left Foot in the air")
            elif self.gait_index==3 and self._on_press("right"):
                self.jump_air_leg = 1
                self.gait_index = 4
                print("Right Foot in the air")


        # Velocity_level Change
        if self._on_press("select"):
            self.commands_level = self.low_speed_level.copy()
            print("LOW SPEED")
        if self._on_press("start"):
            self.commands_level = self.high_speed_level.copy()
            print("HIGH SPEED")
            
        # Clip Command
        if self.gait_index == 0: # Standing
            self.commands[:3] = 0
        elif self.gait_index == 2: # Hopping
            self.commands[6] = min(self.commands[6], 0.2) # max swing height
            self.commands[8] = min(self.commands[8], 0.3) # max pitch orientation
        elif self.gait_index == 3 or self.gait_index == 4:
            self.commands[0] = np.clip(self.commands[0], -0.5, 0.5)
            self.commands[1] = np.clip(self.commands[1], -0.5, 0.5)
            self.commands[2] = np.clip(self.commands[2], -0.5, 0.5)
            self.commands[3] = np.clip(self.commands[3], 0.8, 1.5)
            self.commands[6] = np.clip(self.commands[6], 0.1, 0.2)
            self.commands[7] = np.clip(self.commands[7], -0.35, 0.0)
            self.commands[8:] = 0

        self.velocity_level = np.abs(self.commands[0]) + np.abs(self.commands[1]) + 0.5 * np.abs(self.commands[2])
        # if self.velocity_level > 1.8:
        #     self.commands[3] = max(self.commands[3], 2.0) # frequency
        #     self.commands[8] = min(self.commands[8], 0.3) # body_pitch
        #     self.commands[9] = max(self.commands[9], -0.15) # waist rotation
        #     self.commands[9] = min(self.commands[9], 0.15)
        
        if self.commands[7] < -0.2:
            self.commands[6] = min(self.commands[6], 0.2) # swing height
            self.commands[8] = min(self.commands[8], 0.3) # body_pitch
        
        if self.commands[3] > 2.5:
            self.commands[6] = min(self.commands[6], 0.2) # swing height

        #print(self.commands) 
        
    def init(self):
        self.set_weight(0.0) 
        self.set_to_target_pos(self.init_joint_pos)
        time.sleep(0.1)
        self.set_weight(1.0)
        self.set_to_target_pos(self.init_joint_pos)
        time.sleep(0.1)

    def set_to_target_pos(self, target_pos=None):
        current_jpos_des = self.get_joint_pos()
        if target_pos is None:
            target_pos = self.init_joint_pos
        for i in range(500):
            for j in range(len(target_pos)):
                current_jpos_des[j] += np.clip(
                    target_pos[j] - current_jpos_des[j],
                    -self.max_joint_delta,
                    self.max_joint_delta,
                )

            self.step( 1/ACTION_SCALE * (current_jpos_des-self.default_pos) )
            time.sleep(0.005)

    def get_lower_joint(self):
        motor_state = self.get_state()
        return np.array(
            [motor_state[LowerJoints[j]].q for j in range(len(LowerJoints))]
        )

    def get_default_pos(self):
        return self.default_pos #For policy.
    
    def extract_joint_pos(self, motor_state):
        return np.array(
            [motor_state[WholeBodyJoints[j]].q for j in range(len(WholeBodyJoints))]
        )
    
    def extract_joint_vel(self, motor_state):
        return np.array(
            [motor_state[WholeBodyJoints[j]].dq for j in range(len(WholeBodyJoints))]
        )
    
    
    def get_joint_pos(self):
        motor_state = self.get_state()
        return self.extract_joint_pos(motor_state)

    def set_weight(self, weight):
        self.msg.motor_cmd[JointIndex["kNotUsedJoint"]].q = weight

    def step(self, action=None, dq=0.0, tau_ff=0.0):  # action = target_joints  
        
        # set control joints
        if action is None:
            self.send_msg()
            return
        
        self.last_action = action.copy()
        for j in range(len(action)):
            self.msg.motor_cmd[WholeBodyJoints[j]].q = action[j] * ACTION_SCALE + self.default_pos[j]
            self.msg.motor_cmd[WholeBodyJoints[j]].dq = dq
            self.msg.motor_cmd[WholeBodyJoints[j]].tau = tau_ff

        self.send_msg()
    
    
    def set_motor_gains(self):
        for  k,v in JointIndex.items():
            if k in ['kRightHipYaw', 'kRightHipRoll', 'kRightHipPitch', 'kLeftHipYaw', 'kLeftHipRoll', 'kLeftHipPitch']:
                self.msg.motor_cmd[v].kp = kp_hip
                self.msg.motor_cmd[v].kd = kd_hip
                self.msg.motor_cmd[v].mode = 0x0A
            elif k in ['kRightKnee', 'kLeftKnee', 'kWaistYaw']:
                self.msg.motor_cmd[v].kp = kp_knee_torso
                self.msg.motor_cmd[v].kd = kd_knee_torso
                self.msg.motor_cmd[v].mode = 0x0A
            elif k in ['kLeftAnkle', 'kRightAnkle']:
                self.msg.motor_cmd[v].kp = kp_ankle
                self.msg.motor_cmd[v].kd = kd_ankle
                self.msg.motor_cmd[v].mode = 0x01
            elif k in ['kRightShoulderPitch', 'kRightShoulderRoll', 'kRightShoulderYaw', 'kRightElbow', 'kLeftShoulderPitch', 'kLeftShoulderRoll', 'kLeftShoulderYaw', 'kLeftElbow']:
                self.msg.motor_cmd[v].kp = kp_arm
                self.msg.motor_cmd[v].kd = kd_arm
                self.msg.motor_cmd[v].mode = 0x01
            
    def compute_obs(self):
        msg = self.sub.Read()
        
        self.WirelessControlFrom_msg(msg.wireless_remote)
        if self.Control_Mode.value == 4:
            self.calculate_predefined_commands()
        else:
            self.state_machine()
            # self.joystick_buffer.append(deepcopy(self.key_state))


            
        # clip commands
        for i in range(3):
            if abs(self.commands[i]) < 0.2:
                self.commands[i] = 0.0
            self.tgt_commands[i] = cmd2out1step(self.commands[i], self.tgt_commands[i], self.control_dt, 3.0, 0.6)
        # print(self.tgt_commands)
        obs = np.concatenate(
            (
                np.array(msg.imu_state.gyroscope) * obs_scale.ang_vel,
                quat_rotate_inverse(np.array(msg.imu_state.quaternion), self.gravity_vec),
                (self.extract_joint_pos(msg.motor_state)-self.default_pos) * obs_scale.dof_pos,
                self.extract_joint_vel(msg.motor_state) * obs_scale.dof_vel,
                self.last_action,
                self.tgt_commands * obs_scale.commands[:3],
                self.commands[3:] * obs_scale.commands[3:],
            ), axis = -1
            )
        return obs
    
    def get_gait_idx(self):
        return self.gait_index
    

    def get_state(self):
        msg = self.sub.Read()
        return msg.motor_state
    
    def get_joystick_state(self):
        msg = self.sub.Read()
        return msg.wireless_remote
    
    def get_imu_state(self):
        msg = self.sub.Read()
        return msg.imu_state.rpy, msg.imu_state.gyroscope

    def get_joint_vel(self):
        motor_state = self.get_state()  
        return self.extract_joint_vel(motor_state)

    def send_msg(self):
        self.msg.crc = self.crc.Crc(self.msg)
        self.pub.Write(self.msg)

    def finalize(self):
        # file_name = "../joystick_logs/"
        # if not os.path.exists(file_name):
        #     os.makedirs(file_name)
        # cnt = 0
        # while os.path.exists(os.path.join(file_name, f"{cnt}.npy")):
        #     cnt += 1
        # np.save(os.path.join(file_name, f"{cnt}.npy"), self.joystick_buffer)
        self.set_to_target_pos(self.final_joint_pos)
        self.set_weight(0.0)
        self.set_to_target_pos(self.final_joint_pos)

        for k,v in JointIndex.items():
            self.msg.motor_cmd[v].kp = 0
            self.msg.motor_cmd[v].kd = 0
            self.msg.motor_cmd[v].mode=0x00
        self.send_msg()
        self.pub.Close()


class H1_interrupt(H1):
    def __init__(self, Control_Mode, interrupt_flag, keys_share, control_dt=0.02, max_joint_velocity=0.5, weight_rate=0.2) -> None:
        super().__init__(Control_Mode, interrupt_flag, keys_share, control_dt, max_joint_velocity, weight_rate)
        self.commands = np.array([0, 0, 0, 
                                  2.0, 0.5, 0.5,  # Frequency, phases, duration
                                  0.15, 0.0,      # swing_height body_height
                                  0.0, 0.0, 0.0], dtype=np.float32)
    def state_machine(self):
        super().state_machine()
        if self._pressed("R2"):
            if self._on_press("Y"):
                self.interrupt_flag.value = 1
                self.commands[10] = 1
                print("Start Interruption")
            elif self._on_press("X"):
                self.interrupt_flag.value = 2
                self.commands[10] = 1
                print("Second Interruption")
            elif self._on_press("A"):
                self.interrupt_flag.value = 0
                self.commands[10] = 0 
                print("Stop Interruption")
        if self.commands[10]:
           self.commands[0] = np.clip(self.commands[0], a_min=-0.6, a_max=0.6)

    def compute_obs(self):
        # st = time.time()
        msg = self.sub.Read()
        self.WirelessControlFrom_msg(msg.wireless_remote)
        # print("Read Time:", time.time()-st) 
        if self.Control_Mode.value == 4:
            self.calculate_predefined_commands()
        else:
            self.state_machine()
            # self.joystick_buffer.append(deepcopy(self.key_state))

        for i in range(3):
            # if abs(self.commands[i]) < 0.2:
                # self.commands[i] = 0.0
            self.tgt_commands[i] = cmd2out1step(self.commands[i], self.tgt_commands[i], self.control_dt, 3.0, 0.6)
        # print(self.tgt_commands)
        dof_pos = (self.extract_joint_pos(msg.motor_state)-self.default_pos) * obs_scale.dof_pos 
        dof_vel = self.extract_joint_vel(msg.motor_state) * obs_scale.dof_vel
        # A = time.time()
        if MASK_INT and self.interrupt_flag.value:
            dof_pos[-8:] = 0
            dof_vel[-8:] = 0
        # self.commands[-1] = 0

        ang_vel = msg.imu_state.gyroscope
        # B = time.time()
        # ang_vel = np.array(msg.imu_state.gyroscope)
        quat = msg.imu_state.quaternion

        waist_yaw = msg.motor_state[JointIndex["kWaistYaw"]].q
        waist_yaw_omega = msg.motor_state[JointIndex["kWaistYaw"]].dq
        # print(waist_yaw, waist_yaw_omega)
        gravity_vec, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)
        # print(gravity_vec)
        # ang_vel = ang_vel[0]
        # gravity_vec = get_gravity_orientation(quat)
        # print(ang_vel.shape, gravity_vec.shape)
        # C = time.time()
        obs = np.concatenate(
            (
                ang_vel * obs_scale.ang_vel,
                gravity_vec,
                dof_pos,
                dof_vel,
                self.last_action,
                self.tgt_commands * obs_scale_interrupt.commands[:3],
                self.commands[3:] * obs_scale_interrupt.commands[3:],
            ), axis = -1
        )
        # D = time.time()
        # print("A:", A-st, "B:", B-A, "C:", C-B, "D:", D-C)
        # print("OBS:", obs.shape)
        # print("ang_vel shape:", ang_vel.shape)
        # print("gravity_vec shape:", gravity_vec.shape)
        # print("dof_pos shape:", dof_pos.shape)
        # print("dof_vel shape:", dof_vel.shape)
        # print("last_action shape:", self.last_action.shape)
        # print("tgt_commands shape:", self.tgt_commands.shape)
        # print("commands shape:", self.commands[3:].shape)
        return obs

    
    
    def step(self, action=None, interrupt_action=None, dq=0.0, tau_ff=0.0):  # action = target_joints  
        if self.Control_Mode.value!=2 and self.Control_Mode.value!=4:
            self.WirelessControlFrom_msg(self.get_joystick_state())
    
        # set control joints
        if action is None:
            self.send_msg()
            return
        
        self.last_action[:] = action
        for j in range(11):
        # for j in range(len(action)):
            self.msg.motor_cmd[WholeBodyJoints[j]].q = action[j] * ACTION_SCALE + self.default_pos[j]
            self.msg.motor_cmd[WholeBodyJoints[j]].dq = dq
            self.msg.motor_cmd[WholeBodyJoints[j]].tau = tau_ff
        if interrupt_action is not None:

            for j in range(11, 19):
                self.msg.motor_cmd[WholeBodyJoints[j]].q = interrupt_action[j-11] * ACTION_SCALE + self.default_pos[j]
                self.msg.motor_cmd[WholeBodyJoints[j]].dq = dq
                self.msg.motor_cmd[WholeBodyJoints[j]].tau = tau_ff
                self.msg.motor_cmd[WholeBodyJoints[j]].kp = 80 #kp_arm #100
                self.msg.motor_cmd[WholeBodyJoints[j]].kd = 1 #kd_arm #2
        else:
            for j in range(11, 19):
                self.msg.motor_cmd[WholeBodyJoints[j]].q = action[j] * ACTION_SCALE + self.default_pos[j]
                self.msg.motor_cmd[WholeBodyJoints[j]].dq = dq
                self.msg.motor_cmd[WholeBodyJoints[j]].tau = tau_ff
                self.msg.motor_cmd[WholeBodyJoints[j]].kp = kp_arm
                self.msg.motor_cmd[WholeBodyJoints[j]].kd = kd_arm

        self.send_msg()

if __name__ == "__main__":
    weight = 0.0
    weight_rate = 0.2

    control_dt = 0.02
    max_joint_velocity = 0.5

    delta_weight = weight_rate * control_dt
    max_joint_delta = max_joint_velocity * control_dt
    sleep_time = control_dt / 0.001 / 1000 / 4

    init_time = 5.0 * 2
    init_time_steps = int(init_time / control_dt)

    h1 = H1(
        control_dt=control_dt,
        max_joint_velocity=max_joint_velocity,
        weight_rate=weight_rate,
    )

    joint_pos = h1.get_joint_pos()
    h1.reset_to_init()
    h1.finalize()
