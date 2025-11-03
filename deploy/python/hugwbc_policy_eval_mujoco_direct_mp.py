from h1_mujoco_direct import H1MujocoDirect
from h1_wbc_policy import GetAction
from h1_wbc_policy_interrupt import interrupt_sampler
from consts import *
import time
from multiprocessing import Process, Array, Value
import torch
import os
import numpy as np
import ctypes


# Match interrupt setup
OBS_DIM = 76
ACT_DIM = 19
INTERRUPT_DIM = 8
CONTROL_DT = 0.02
MAX_JOINT_VELOCITY = 0.2
WEIGHT_RATE = 0.2


def ControlRobotDirect(shared_obs_base, shared_act_base, executed_interrupt_base, gait_idx, interrupt_flag, Control_Mode, key_share):
    obs = np.frombuffer(shared_obs_base, dtype=ctypes.c_float)
    act = np.frombuffer(shared_act_base, dtype=ctypes.c_float)
    executed_interrupt = np.frombuffer(executed_interrupt_base, dtype=ctypes.c_float)

    print("Start Direct Mujoco Control")
    robot = H1MujocoDirect(Control_Mode, interrupt_flag, key_share, control_dt=CONTROL_DT, max_joint_velocity=MAX_JOINT_VELOCITY, weight_rate=WEIGHT_RATE)
    print("H1MujocoDirect Initialize finished")
    obs[:-2] = robot.compute_obs()
    gait_idx.value = robot.get_gait_idx()
    default_pos = robot.get_default_pos()
    last_control_mode = 0
    while Control_Mode.value != 3:
        st = time.time()
        control_mode = Control_Mode.value
        if control_mode == 0:
            robot.step()
        elif control_mode == 1:
            if last_control_mode != 1:
                print("Start to Default")
                robot.set_to_target_pos(default_pos)
            robot.step()
        elif control_mode == 2 or control_mode == 4:
            if interrupt_flag:
                robot.step(act, interrupt_action=executed_interrupt)
            else:
                robot.step(act)
        last_control_mode = control_mode
        obs[:-2] = robot.compute_obs()
        gait_idx.value = robot.get_gait_idx()
        slp = CONTROL_DT - (time.time() - st)
        if slp > 0:
            time.sleep(slp)

    robot.finalize()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    shared_obs_base = Array(ctypes.c_float, OBS_DIM, lock=False)
    shared_act_base = Array(ctypes.c_float, ACT_DIM, lock=False)
    shared_interrupt_base = Array(ctypes.c_float, INTERRUPT_DIM, lock=False)
    executed_interrupt_base = Array(ctypes.c_float, INTERRUPT_DIM, lock=False)
    gait_index = Value(ctypes.c_int, lock=False)
    Control_Mode = Value(ctypes.c_byte, lock=False)
    interrupt_flag = Value(ctypes.c_byte, lock=False)
    key_share = Array(ctypes.c_uint8, 40, lock=False)

    p1 = Process(target=GetAction, args=(shared_obs_base, shared_act_base, shared_interrupt_base, executed_interrupt_base, gait_index, interrupt_flag, Control_Mode))
    p2 = Process(target=ControlRobotDirect, args=(shared_obs_base, shared_act_base, executed_interrupt_base, gait_index, interrupt_flag, Control_Mode, key_share))
    p3 = Process(target=interrupt_sampler, args=(shared_interrupt_base, Control_Mode, interrupt_flag, ))

    p1.start()
    p2.start()
    p3.start()
    os.system("sudo renice -20 -p %s " % p1.pid)
    os.system("sudo renice -20 -p %s " % p2.pid)
    os.system("sudo renice -20 -p %s " % p3.pid)
    p1.join()
    p2.join()
    p3.join()


