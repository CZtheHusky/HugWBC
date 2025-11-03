from h1_whole_body_control import H1_interrupt
from h1_wbc_policy import H1_Policy
from h1_wbc_policy_interrupt import interrupt_sampler
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
OBS_DIM=76
ACT_DIM=19
INTERRUPT_DIM=8
MAX_JOINT_VELOCITY=0.2
WEIGHT_RATE=0.2
HIDDEN_DIM = 256
LAYER_NUM = 3
POLICY_DT = 0.02
CONTORL_DECIMATION = 8

CONTROL_DT = POLICY_DT / CONTORL_DECIMATION
EPSDT = 0.00012
freq_dim = 66 # the 66th dim of obs is the command frequency
phase_dim = 67 # the 67th dim of obs is the command delta phase
clip_rad = 0.06 # clip_rad for interruption

JIT_MODEL_PATH = '/cpfs/user/caozhe/workspace/HugWBC/logs/h1_interrupt/exported/policies/trace_jit.pt'
ONNX_MODEL_PATH = '../checkpoints/net.onnx'
DEVICE = 'cpu'#'cuda'

def MainControlProcess(executed_interrupt_base, interrupt_flag, Control_Mode, keys_share):
    # merge the GetAction and Control Robot into 1 process.
    robot = H1_interrupt(Control_Mode, interrupt_flag, keys_share, control_dt=CONTROL_DT, max_joint_velocity=MAX_JOINT_VELOCITY, weight_rate=WEIGHT_RATE)
    executed_interrupt = np.frombuffer(executed_interrupt_base, dtype=ctypes.c_float)

    count = 0
    last_control_mode = 0
    default_pos = robot.get_default_pos()
    global_phase = 0
    policy =  H1_Policy(model_path=JIT_MODEL_PATH, device=DEVICE)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[:-2] = robot.compute_obs()
    obs[-2:] = 0
    act = policy.take_action(obs)
    policy.reset()
    while Control_Mode.value!=3:
        last_time = time.time()
        count += 1
        control_mode = Control_Mode.value
        # print("CONTROL_MODE:", control_mode)
        gait_index = robot.gait_index
        if control_mode == 0:
            robot.step()
        elif control_mode == 1:
            if last_control_mode!=1:
                # Change to Default
                print("Start to Default")
                robot.set_default_pos(default_pos)
            robot.step()
        elif control_mode == 2 or control_mode ==4:
            if count==CONTORL_DECIMATION or (last_control_mode!=2 and last_control_mode!=4): # Update Obs, and calculate the action
                obs[:-2] = robot.compute_obs()
                # obs_time = time.time()
                # print("OBS_TIME:", obs_time-last_time)
                # print(obs)
                act = policy.take_action(obs)
                # act_time = time.time()
                # print("Inference: ", act_time-obs_time)
            if interrupt_flag.value:
                robot.step(act, interrupt_action=executed_interrupt)
            else:
                robot.step(act)
            # print("ACTTIME_B:", time.time()-last_time)
            # UPDATE CLOCK
            if count==CONTORL_DECIMATION or (last_control_mode!=2 and last_control_mode!=4):
                # test_time = time.time()
                count = 0
                global_phase += robot.commands[3] * POLICY_DT
                global_phase %= 1
                left_phase = global_phase + robot.commands[4]
                right_phase = global_phase

                if gait_index == 0:
                    left_clock = 0
                    right_clock = 0
                elif gait_index == 1 or gait_index == 2:
                    left_clock = np.sin(2*np.pi * left_phase)
                    right_clock = np.sin(2*np.pi * right_phase)
                elif gait_index == 3:
                    left_clock = -1
                    right_clock = np.sin(2*np.pi * right_phase)
                elif gait_index == 4:
                    left_clock = np.sin(2*np.pi * left_phase)
                    right_clock = -1
                obs[-2] = left_clock
                obs[-1] = right_clock
                # print("CLOCK:", left_clock, right_clock)
                # print("CLOCK_TIME:", time.time()-test_time)
        last_control_mode = control_mode
        # print("FREQ:", 1/(time.time()-last_time))
        slptm = CONTROL_DT - time.time() + last_time
        if slptm>0:
            time.sleep(slptm)

    robot.finalize()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    shared_interrupt_base = Array(
        ctypes.c_float, INTERRUPT_DIM, lock=False
    )
    Control_Mode = Value(
        ctypes.c_byte, lock=False
    )
    interrupt_flag = Value(
        ctypes.c_byte, lock=False
    )
    # shared wireless controller buffer (40 bytes)
    keys_share = Array(
        ctypes.c_uint8, 40, lock=False
    )
    p1 = Process(target=MainControlProcess, args=(shared_interrupt_base, interrupt_flag, Control_Mode, keys_share,))
    p2 = Process(target=interrupt_sampler, args=(shared_interrupt_base, Control_Mode, interrupt_flag, ))
    p1.start()
    p2.start()
    os.system("sudo renice -20 -p %s " % p1.pid)
    os.system("sudo renice -20 -p %s " % p2.pid)
    p1.join()
    p2.join()