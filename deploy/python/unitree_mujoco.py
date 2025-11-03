import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_mujoco_bridge import UnitreeSdk2Bridge, ElasticBand

import mujoco_config as mujoco_config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(mujoco_config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if mujoco_config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if mujoco_config.ROBOT == "h1" or mujoco_config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

# default tracking target body (for camera lookat)
try:
    if mujoco_config.ROBOT == "h1" or mujoco_config.ROBOT == "g1":
        _track_body_id = mj_model.body("torso_link").id
    else:
        _track_body_id = mj_model.body("base_link").id
except Exception:
    _track_body_id = 0

mj_model.opt.timestep = mujoco_config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(mujoco_config.DOMAIN_ID, mujoco_config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if mujoco_config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=mujoco_config.JOYSTICK_TYPE)
    if mujoco_config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if mujoco_config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        # Update camera to track the target body position
        try:
            p = mj_data.xpos[_track_body_id]
            viewer.cam.lookat[:] = p
        except Exception:
            pass
        viewer.sync()
        locker.release()
        time.sleep(mujoco_config.VIEWER_DT)
        
"""
vglrun -d egl python unitree_mujoco.py
"""

if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
