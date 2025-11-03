from h1_whole_body_control import H1_interrupt
from h1_wbc_policy import GetAction
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
CONTROL_DT=0.02
MAX_JOINT_VELOCITY=0.2
WEIGHT_RATE=0.2
HIDDEN_DIM = 256
LAYER_NUM = 3

freq_dim = 66 # the 66th dim of obs is the command frequency
phase_dim = 67 # the 67th dim of obs is the command delta phase
clip_rad = 0.06 # clip_rad for interruption

JIT_MODEL_PATH = '/cpfs/user/caozhe/workspace/HugWBC/logs/h1_interrupt/exported/policies/trace_jit.pt'
ONNX_MODEL_PATH = '../checkpoints/net.onnx'
DEVICE = 'cuda'#'cuda'

def JointPosInterpolation(initPos, targetPos, rate):
    rate = min(rate, 1)
    Pos = initPos * (1 - rate) + targetPos * rate

    return Pos


def ControlRobot(shared_obs_base, shared_act_base, executed_interrupt_base, gait_idx, interrupt_flag, Control_Mode, key_share):
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_float)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_float)
    executed_interrupt = np.frombuffer(executed_interrupt_base, dtype=ctypes.c_float)

    print("Start Control")
    robot = H1_interrupt(Control_Mode, interrupt_flag, key_share, control_dt=CONTROL_DT, max_joint_velocity=MAX_JOINT_VELOCITY, weight_rate=WEIGHT_RATE)
    print("H1 Initialize finished")
    obs[:-2] = robot.compute_obs()
    gait_idx.value = robot.get_gait_idx()
    default_pos = robot.get_default_pos()
    last_control_mode = 0
    count = 0
    while Control_Mode.value != 3:
        st = time.time()
        control_mode = Control_Mode.value
        # print(control_mode, last_control_mode)
        if control_mode == 0:
            robot.step()
        elif control_mode == 1:
            if last_control_mode!=1:
                # Change to Default
                print("Start to Default")
                robot.set_to_target_pos(default_pos)
            # Step Last Pos
            robot.step()
        elif control_mode==2 or control_mode == 4:
            # Step Action
            if interrupt_flag:
                robot.step(act, interrupt_action=executed_interrupt)
            else:
                robot.step(act)
        
        last_control_mode = control_mode
        obs[:-2] = robot.compute_obs()
        # print("OBS:TIME: ", time.time())
        gait_idx.value = robot.get_gait_idx()
        # print("STEP FREQ:", 1/(time.time()-st))

    robot.finalize()

def interrupt_sampler(shared_interrupt_base, Control_Mode, interrupt_flag):
    target_pos = np.frombuffer(shared_interrupt_base, dtype=ctypes.c_float)
    length = len(Interrupt_Commands)
    last_flag = False
    current_stage = 0
    while Control_Mode.value != 3:
        current_flag = interrupt_flag.value
        if current_flag:
            if last_flag != current_flag:
                # Start to interrupt:
                current_stage = 0
                if current_flag == 1:
                    length = len(Interrupt_Commands)
                elif current_flag == 2:
                    length = len(Interrupt_Commands2)
                # pass
            else:
                if current_stage == length-1:
                    pass
                else:
                    current_stage =(current_stage + 1)

            print("Int Idx: ", current_stage)
            if current_flag == 1:
                target_pos[:] = Interrupt_Commands[current_stage]["Commands"]
                print("target_pos: ", target_pos[3], target_pos[7])
                time.sleep(Interrupt_Commands[current_stage]["Interval"])
            elif current_flag == 2:
                target_pos[:] = Interrupt_Commands2[current_stage]["Commands"]
                print("target_pos: ", target_pos[3], target_pos[7])
                time.sleep(Interrupt_Commands2[current_stage]["Interval"])
        else:
            time.sleep(0.02) # Waiting
        last_flag = current_flag
    
    # while Control_Mode.value != 3:
    #     current_flag = interrupt_flag.value
    #     if current_flag:
    #         if not last_flag:
    #             # Start to interrupt:
    #             current_stage = 0
    #             # pass
    #         else:
    #             current_stage =(current_stage + 1)%length
    #         print("Int Idx: ", current_stage)
    #         target_pos[:] = Interrupt_Commands[current_stage]["Commands"]
    #         print("target_pos: ", target_pos[3], target_pos[7])
    #         time.sleep(Interrupt_Commands[current_stage]["Interval"])
    #     else:
    #         time.sleep(0.02) # Waiting
    #     last_flag = current_flag
    
def Save_Joymsg(Control_Mode, key_share, shared_obs_base):
    keys = np.frombuffer(key_share, dtype=ctypes.c_uint8)
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_float)
    list_key_msgs = []
    list_obs = []
    while Control_Mode.value != 3:
        st = time.time()
        list_key_msgs.append(deepcopy(keys))
        list_obs.append(deepcopy(obs))
        time.sleep(0.02-(time.time()-st))
    cnt = 0
    if not os.path.exists("../joystick_logs"):
        os.makedirs("../joystick_logs")
    while os.path.exists(f"../joystick_logs/joy_msgs_{cnt}.npy"):
        cnt+=1
    np.save(f"../joystick_logs/joy_msgs_{cnt}.npy", np.array(list_key_msgs))
    np.save(f"../joystick_logs/obs_{cnt}.npy", np.array(list_obs))

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
    key_share = Array(
        ctypes.c_uint8, 40, lock=False
    )
    # Control_Finish = Event()
    # Switch_Default = Event()
    # Start_Control = Event()
    # Standing = Event()

    p1 = Process(target=GetAction, args=(shared_obs_base, shared_act_base, shared_interrupt_base, executed_interrupt_base, gait_index, interrupt_flag, Control_Mode))
    p2 = Process(target=ControlRobot, args=(shared_obs_base, shared_act_base, executed_interrupt_base, gait_index, interrupt_flag, Control_Mode, key_share))
    p3 = Process(target=interrupt_sampler, args=(shared_interrupt_base, Control_Mode, interrupt_flag, ))
    p4 = Process(target=Save_Joymsg, args=(Control_Mode, key_share, shared_obs_base))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    os.system("sudo renice -20 -p %s " % p1.pid)
    os.system("sudo renice -20 -p %s " % p2.pid)
    os.system("sudo renice -20 -p %s " % p3.pid)
    os.system("sudo renice -20 -p %s " % p4.pid)
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    

