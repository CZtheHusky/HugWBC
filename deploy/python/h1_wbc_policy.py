from h1_whole_body_control import H1
from consts import *
import time
from multiprocessing import Process, Array, Event, Value
import torch
import os
import sys
import numpy as np
import ctypes
import copy
# import onnxruntime

# OBS_DIM=66
#OBS_DIM=76
OBS_DIM=75
ACT_DIM=19
INTERRUPT_DIM=8
CONTROL_DT=0.02
MAX_JOINT_VELOCITY=0.2
WEIGHT_RATE=0.2
HIDDEN_DIM = 256
LAYER_NUM = 3

freq_dim = 66 # the 66th dim of obs is the command frequency
phase_dim = 67 # the 67th dim of obs is the command delta phase
clip_rad = 0.1 # clip_rad for interruption

JIT_MODEL_PATH = '/cpfs/user/caozhe/workspace/HugWBC/logs/h1_interrupt/exported/policies/trace_jit.pt'
ONNX_MODEL_PATH = '../checkpoints/net.onnx'
DEVICE = 'cpu'#'cuda'
LOGS_PATH = "../logs"


class H1_Policy:
    def __init__(
        self,
        model_path: str,
        device: str,
    ):
        '''
        model_path: str, Path of jit model. 
        device: str, cuda or cpu.
        '''
        self.jit_policy = torch.jit.load(model_path, map_location=device)
        self.device = device
        self.jit_policy.to(device)
        tmp = 0
        while os.path.exists(os.path.join(LOGS_PATH, "experiment_%d" % tmp)):
            tmp += 1
        self.log_path = os.path.join(LOGS_PATH, "experiment_%d" % tmp)
        os.makedirs(self.log_path)
        self.counts = 0

        self.obs_buffer = []
        self.act_buffer = []

    def __call__(self, obs: torch.tensor) -> torch.tensor:
        '''
        Input:
            obs: torch.tensor, should be in the same device as policy. (T, BS, OBS_DIM, )
        Output:
            action: torch.tensor, in the same device as policy, (B, ACT_DIM, )
        '''
        # print(obs.shape, obs.type)
        # act, _ = self.jit_policy(obs) # For RNN Module with output hidden.
        # print(obs.shape)
        act = self.jit_policy(obs)
        # self.obs_buffer.append(obs)
        # self.act_buffer.append(act)
        
        return act # Only return the action part.

    def save_buffered_data(self):
        if len(self.obs_buffer)==0:
            return
        obs_batch = torch.cat(self.obs_buffer, dim=0)
        act_batch = torch.cat(self.act_buffer, dim=0)
        
        torch.save(obs_batch, os.path.join(self.log_path, f"obs_{self.counts}.pt"))
        torch.save(act_batch, os.path.join(self.log_path, f"act_{self.counts}.pt"))
        
        self.obs_buffer = []
        self.act_buffer = []
        self.counts += 1
    
    def take_action(self, obs: np.ndarray) -> np.ndarray:
        '''
        Input:
            obs: np.ndarray, load from share memory or sth like that. (OBS_DIM, )
        Output:
            action: np.ndarray, will be written to share memory or sth like that. (ACT_DIM, )
        '''
        try:
            return self.__call__(torch.FloatTensor(obs).to(self.device).reshape(1,1,-1)).detach().cpu().numpy().squeeze()
        except:
            try:
                return self.__call__(torch.FloatTensor(obs).to(self.device).reshape(1,-1)).cpu().numpy().squeeze()
            except:
                return self.__call__(torch.FloatTensor(obs).to(self.device).reshape(-1))[0].cpu().numpy().squeeze()

    
    def reset(self):
        try:
            self.jit_policy.reset_memory(torch.zeros(1,76).to(self.device))
        except:
            self.jit_policy.reset_memory()
        self.counts = 0
        self.obs_buffer = []
        self.act_buffer = []

def GetAction(shared_obs_base, shared_act_base, shared_interrupt_base, executed_interrupt_base, gait_index, interrupt_flag, Control_Mode):
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_float)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_float)
    target_pos = np.frombuffer(shared_interrupt_base, dtype=ctypes.c_float)
    executed_interrupt = np.frombuffer(executed_interrupt_base, dtype=ctypes.c_float)
    # gait_idx = np.frombuffer(shaerd_gait_base, dtype=ctypes.c_int)
    
    global_phase = 0
    control_dt = CONTROL_DT
    last_gait = 0
    stand_switching = 0 # 0 for totally standing.
    jump_switching = 0 # Default Left

    policy =  H1_Policy(model_path=JIT_MODEL_PATH, device=DEVICE)
    
    # policy warm up
    for _ in range(5):
        policy.take_action(obs)
    policy.reset()
    counts = 0
    with torch.inference_mode():
        while Control_Mode.value != 3:
            last_time = time.time()
            if Control_Mode.value != 2 and Control_Mode.value != 4:
                time.sleep(0.1)
                # Calc Gait Clock OBS
                gait_idx = gait_index.value
                global_phase += (obs[freq_dim] * control_dt)
                global_phase = np.remainder(global_phase, 1.0)
                left_clock_phase = (global_phase + obs[phase_dim])
                right_clock_phase = global_phase

                if gait_idx == 0: #Standing    
                    stand_switching = max(0, stand_switching-0.1)
                else: # Non-Standing
                    stand_switching = min(1, stand_switching+0.1)
                if gait_idx != 3 and gait_idx!=4: jump_switching = 0 
                if gait_idx == 3: # Jumping, Left in the air
                    if last_gait == 4:
                        jump_switching = 1
                    jump_switching = max(0, jump_switching - 0.1)
                    left_clock_phase = 0.75 * (1-jump_switching) + 0.5 * jump_switching # interpolate from ground to air
                    right_clock_phase = 0.75 * jump_switching + right_clock_phase * (1-jump_switching) #interpolate from air to the sin phase.
                elif gait_idx == 4: # Jumping, Right in the air
                    if last_gait == 3:
                        jump_switching = 1
                    jump_switching = max(0, jump_switching - 0.1)
                    left_clock_phase = 0.75 * jump_switching + left_clock_phase * (1-jump_switching)
                    right_clock_phase = 0.75 * (1-jump_switching) + 0.5 * jump_switching
                last_gait = gait_idx
                left_clock = stand_switching * np.sin(left_clock_phase*2*np.pi)
                right_clock = stand_switching * np.sin(right_clock_phase*2*np.pi)

                obs[-2] = left_clock
                obs[-1] = right_clock
                continue

            # Take Action
            obs[-2]=left_clock
            obs[-1]=right_clock
            # print("IN GETACTION gait idx:", gait_idx)
            act[:] = policy.take_action(obs)
            # print("ACTION: ", time.time())
            if interrupt_flag.value: # Interrupt
                executed_interrupt[:] = target_pos[:]
                # executed_interrupt[:] = np.clip(
                #     target_pos,
                #     (-clip_rad + obs[17:25]) * 4,
                #     (clip_rad + obs[17:25]) * 4
                # )
            
            
            # Calculate Clock at target frequency.
            gait_idx = gait_index.value
            global_phase +=(obs[freq_dim] * control_dt)
            global_phase = np.remainder(global_phase, 1.0)
            left_clock_phase = (global_phase + obs[phase_dim])
            right_clock_phase = global_phase

            if gait_idx == 0: #Standing    
                stand_switching = max(0, stand_switching-0.1)
            else: # Non-Standing
                stand_switching = min(1, stand_switching+0.1)
            if gait_idx != 3 and gait_idx!=4: jump_switching = 0 
            if gait_idx == 3: # Jumping, Left in the air
                if last_gait == 4:
                    jump_switching = 1
                jump_switching = max(0, jump_switching - 0.1)
                left_clock_phase = 0.75 * (1-jump_switching) + 0.5 * jump_switching # interpolate from ground to air
                right_clock_phase = 0.75 * jump_switching + right_clock_phase * (1-jump_switching) #interpolate from air to the sin phase.
            elif gait_idx == 4: # Jumping, Right in the air
                if last_gait == 3:
                    jump_switching = 1
                jump_switching = max(0, jump_switching - 0.1)
                left_clock_phase = 0.75 * jump_switching + left_clock_phase * (1-jump_switching)
                right_clock_phase = 0.75 * (1-jump_switching) + 0.5 * jump_switching
            last_gait = gait_idx
            left_clock = stand_switching * np.sin(2*np.pi * left_clock_phase)
            right_clock = stand_switching * np.sin(2*np.pi * right_clock_phase)
            obs[-2] = left_clock
            obs[-1] = right_clock
            #print(time.time()-last_time)
            # Wait For the target Frequency
            if CONTROL_DT - time.time() + last_time > 0.01:
                time.sleep(CONTROL_DT- time.time() + last_time)
        policy.save_buffered_data()


def ControlRobot(shared_obs_base, shared_act_base, gait_idx, interrupt_flag, Control_Mode):
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_float)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_float)
    print("Start Control")
    robot = H1(Control_Mode, interrupt_flag, control_dt=CONTROL_DT, max_joint_velocity=MAX_JOINT_VELOCITY, weight_rate=WEIGHT_RATE)
    print("H1 Initialize finished")
    obs[:-2] = robot.compute_obs()
    gait_idx.value = robot.get_gait_idx()
    default_pos = robot.get_default_pos()
    last_control_mode = 0
    while Control_Mode.value != 3:
        control_mode = Control_Mode.value
        # print(control_mode, last_control_mode)
        if control_mode == 0:
            robot.step()
        elif control_mode==1:
            if last_control_mode!=1:
                # Change to Default
                print("Start to Default")
                robot.set_to_target_pos(default_pos)
            # Step Last Pos
            robot.step()
        elif control_mode==2 or control_mode==4:
            # Step Action
            robot.step(act)
        
        last_control_mode = control_mode
        obs[:-2] = robot.compute_obs()
        gait_idx.value = robot.get_gait_idx()


    robot.finalize()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    shared_obs_base = Array(
        ctypes.c_float, OBS_DIM, lock=False
    )
    shared_act_base = Array(
        ctypes.c_float, ACT_DIM, lock=False
    )
    shared_interrupt_base = Array(
        ctypes.c_float, INTERRUPT_DIM, lock=False
    )
    executed_interrupt_base = Array(
        ctypes.c_float, INTERRUPT_DIM, lock=False
    )
    # shaerd_gait_base = Array(
    #     ctypes.c_int, 1, lock=False
    # )
    gait_index = Value(
        ctypes.c_int, lock=False
     )
    Control_Mode = Value(
        ctypes.c_byte, lock=False
    )
    interrupt_flag = Value(
        ctypes.c_byte, lock=False
    )
    # Control_Finish = Event()
    # Switch_Default = Event()
    # Start_Control = Event()
    # Standing = Event()

    p1 = Process(target=GetAction, args=(shared_obs_base, shared_act_base, shared_interrupt_base, executed_interrupt_base, gait_index, interrupt_flag, Control_Mode))
    p2 = Process(target=ControlRobot, args=(shared_obs_base, shared_act_base, gait_index, interrupt_flag, Control_Mode))

    p1.start()
    p2.start()
    os.system("sudo renice -20 -p %s " % p1.pid)
    os.system("sudo renice -20 -p %s " % p2.pid)
    p1.join()
    p2.join()

