import numpy as np
from typing import Dict
from copy import copy, deepcopy
USE_SIM = True #True
net_interface = "eth0"
COLLECT_TRAJ_PATH = "../trajectories/react_data_0105_loco"
INTERRUPT_MODE = "Pick" #"Wave" #"Noise" #"Box" #"Pick" #"Box" # "Noise" #"Noise" #"Hello"# "Hello" # "Noise" # "Pick" "Pick2"
MASK_INT = False #True 

kPi = 3.141592654
kPi_2 = 1.57079632

JointIndex = dict(
    # Right leg
    kRightHipYaw=8,
    kRightHipRoll=0,
    kRightHipPitch=1,
    kRightKnee=2,
    kRightAnkle=11,
    # Left leg
    kLeftHipYaw=7,
    kLeftHipRoll=3,
    kLeftHipPitch=4,
    kLeftKnee=5,
    kLeftAnkle=10,
    kWaistYaw=6,
    kNotUsedJoint=9,
    # Right arm
    kRightShoulderPitch=12,
    kRightShoulderRoll=13,
    kRightShoulderYaw=14,
    kRightElbow=15,
    # Left arm
    kLeftShoulderPitch=16,
    kLeftShoulderRoll=17,
    kLeftShoulderYaw=18,
    kLeftElbow=19,
)


ArmJoints = [
    JointIndex["kLeftShoulderPitch"],
    JointIndex["kLeftShoulderRoll"],
    JointIndex["kLeftShoulderYaw"],
    JointIndex["kLeftElbow"],
    JointIndex["kRightShoulderPitch"],
    JointIndex["kRightShoulderRoll"],
    JointIndex["kRightShoulderYaw"],
    JointIndex["kRightElbow"],
]

LowerJoints = [
    JointIndex["kLeftHipYaw"],
    JointIndex["kLeftHipRoll"],
    JointIndex["kLeftHipPitch"],
    JointIndex["kLeftKnee"],
    JointIndex["kLeftAnkle"],
    JointIndex["kRightHipYaw"],
    JointIndex["kRightHipRoll"],
    JointIndex["kRightHipPitch"],
    JointIndex["kRightKnee"],
    JointIndex["kRightAnkle"],
    JointIndex["kWaistYaw"],
]

WholeBodyJoints = LowerJoints + ArmJoints

WeakMotors = [
    JointIndex["kLeftAnkle"],
    JointIndex["kRightAnkle"],
    JointIndex["kRightShoulderPitch"],
    JointIndex["kRightShoulderRoll"],
    JointIndex["kRightShoulderYaw"],
    JointIndex["kRightElbow"],
    JointIndex["kLeftShoulderPitch"],
    JointIndex["kLeftShoulderRoll"],
    JointIndex["kLeftShoulderYaw"],
    JointIndex["kLeftElbow"],
]


def is_weak_motor(joint):
    return joint in WeakMotors or joint in ArmJoints


kp_hip = 200.0
kd_hip = 5.0
kp_knee_torso = 300.0
kd_knee_torso = 6.0
kp_arm = 20
kd_arm = 0.5
kp_ankle = 40
kd_ankle = 2

hip_roll_init_pos_ = 0.02
hip_pitch_init_pos_ = -0.4
knee_init_pos_ = 0.8
ankle_init_pos_ = -0.4
# shoulder_pitch_init_pos_ = 0.4

init_pos = np.array(
            [
                0., hip_roll_init_pos_, hip_pitch_init_pos_,
                knee_init_pos_, ankle_init_pos_,
                0., -hip_roll_init_pos_, hip_pitch_init_pos_,
                knee_init_pos_, ankle_init_pos_,
                0]
        )

default_joint_angles = { # = target angles [rad] when action = 0.0
    'kLeftHipYaw' : 0. ,   
    'kLeftHipRoll' : 0.02,               
    'kLeftHipPitch' : -0.4,         
    'kLeftKnee' : 0.8,       
    'kLeftAnkle' : -0.4,     
    'kRightHipYaw' : 0., 
    'kRightHipRoll' : -0.02, 
    'kRightHipPitch' : -0.4,                                       
    'kRightKnee' : 0.8,                                             
    'kRightAnkle' : -0.4,                                     
    'kWaistYaw' : 0., 
    'kLeftShoulderPitch' : 0., 
    'kLeftShoulderRoll' : 0, 
    'kLeftShoulderYaw' : 0.,
    'kLeftElbow'  : 0.,
    'kRightShoulderPitch' : 0.,
    'kRightShoulderRoll' : 0.0,
    'kRightShoulderYaw' : 0.,
    'kRightElbow' : 0.,
}

key_state = {
    "R1": 0,
    "L1": 0,
    "start": 0,
    "select": 0,
    "R2": 0,
    "L2": 0,
    "F1": 0,
    "F2": 0,
    "A": 0,
    "B": 0,
    "X": 0,
    "Y": 0,
    "up": 0,
    "right": 0,
    "down": 0,
    "left": 0,
}
# class obs_scale:
#     ang_vel = 0.25
#     dof_pos = 1.0
#     dof_vel = 0.05,
#     commands = np.array([2.0, 2.0, 0.25])

class obs_scale:
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05,
    commands = np.array([2.0, 2.0, 0.25,
                         1.0, 1.0, 1.0,  
                         0.15, 2.0, 0.5, 0.5])
                        # 0.15, 2.0])
class obs_scale_interrupt:
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    commands = np.array([2.0, 2.0, 0.25, 1.0, 1.0, 1.0, 0.15, 2.0, 0.5, 0.5, 1])

ACTION_SCALE = 0.25

commands_ranges={ #Default Command Ranges
    'lin_vel_x' : [-0.6, 0.6], # min max [m/s]
    'lin_vel_y' : [-0.6, 0.6],   # min max [m/s]
    'ang_vel_yaw' : [-0.6, 0.6],    # min max [rad/s]
    'gait_frequency': [1.5, 2.5], #[1.5, 3.5]
    'foot_swing_height' : [0.1, 0.5],
    'body_height' :  [-0.35, 0.0],
    'body_pitch' : [0.0, 0.7],
    'waist_roll' : [-1.0, 1.0],

    'limit_vel_x': [-0.6, 2.0],
    'limit_vel_yaw': [-1.0, 1.0], 
}

# Default Command: Standing for 2 seconds.
Default_Standing = {  # 11 dim command.
    "Interval": 3,
    "Commands": np.array([0,0,0,2,0,0.5,0.15,0,0,0,0]),
    "Gait_Index": 0
}

def Raise_Swing_Height(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][6] = min(new_command["Commands"][6]+0.05, commands_ranges['foot_swing_height'][1])
    Keys = ["R1", "Y"]
    return new_command, Keys

def Put_down_Swing_Height(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][6] = max(new_command["Commands"][6]-0.05, commands_ranges['foot_swing_height'][0])
    Keys = ["R1", "A"]
    return new_command, Keys

def Increase_Gait_Frequency(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][3] = min(new_command["Commands"][3]+0.4, commands_ranges['gait_frequency'][1])
    Keys = ["R1", "up"]
    return new_command, Keys

def Decrease_Gait_Frequency(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][3] = max(new_command["Commands"][3]-0.4, commands_ranges['gait_frequency'][0])
    Keys = ["R1", "down"]
    return new_command, Keys

def Turning_Left_Waist(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][9] = min(new_command["Commands"][9] + 0.1, commands_ranges["waist_roll"][1])
    Keys = ["L2", "left"]
    return new_command, Keys

def Turning_Right_Waist(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][9] = max(new_command["Commands"][9] - 0.1, commands_ranges["waist_roll"][0])
    Keys = ["L2", "right"]
    return new_command, Keys


def Turning_Up_Pitch(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][8] = max(new_command["Commands"][8] - 0.05, commands_ranges["body_pitch"][0])
    Keys = ["L2", "Y"]
    return new_command, Keys

def Turning_Down_Pitch(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][8] = min(new_command["Commands"][8] + 0.05, commands_ranges["body_pitch"][1])
    Keys = ["L2", "A"]
    return new_command, Keys

def Increase_Body_Height(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][7] = min(new_command["Commands"][7] + 0.05, commands_ranges["body_height"][1])
    Keys = ["L2", "up"]
    return new_command, Keys

def Decrease_Body_Height(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][7] = max(new_command["Commands"][7] - 0.05, commands_ranges["body_height"][0])
    Keys = ["L2", "down"]
    return new_command, Keys

def Switch_Standing(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Gait_Index"] = 0
    new_command["Commands"][4] = 0
    new_command["Commands"][:3] = 0
    Keys = ["L1", "Y"]
    return new_command, Keys

def Switch_Walking(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Gait_Index"] = 1
    new_command["Commands"][4] = 0.5
    Keys = ["L1", "A"]
    return new_command, Keys

def Switch_Hopping(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Gait_Index"] = 2
    new_command["Commands"][4] = 0
    Keys = ["L1", "B"]
    return new_command, Keys

def Switch_Jumping_Leftair(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Gait_Index"] = 3
    new_command["Commands"][4] = 0
    Keys = ["L1", "X"]
    return new_command, Keys

def Switch_Jumping_Rightair(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Gait_Index"] = 4
    new_command["Commands"][4] = 0
    Keys = ["L1", "X"]
    return new_command, Keys


def Increase_Lin_x(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][0] = min(new_command["Commands"][0]+0.1, commands_ranges["lin_vel_x"][1])
    Keys = ["Lstick up"]
    return new_command, Keys

def Decrease_Lin_x(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][0] = max(new_command["Commands"][0]-0.1, commands_ranges["lin_vel_x"][0])
    Keys = ["Lstick down"]
    return new_command, Keys

def Increase_Lin_y(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][1] = min(new_command["Commands"][1]+0.1, commands_ranges["lin_vel_y"][1])
    Keys = ["Lstick left"]
    return new_command, Keys

def Decrease_Lin_x(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][1] = max(new_command["Commands"][1]-0.1, commands_ranges["lin_vel_y"][0])
    Keys = ["Lstick right"]
    return new_command, Keys

def Increase_Ang_z(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][2] = min(new_command["Commands"][2]+0.1, commands_ranges["ang_vel_yaw"][1])
    Keys = ["Rstick left"]
    return new_command, Keys

def Decrease_Ang_z(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][2] = max(new_command["Commands"][2]-0.1, commands_ranges["ang_vel_yaw"][0])
    Keys = ["Rstick right"]
    return new_command, Keys

def Start_Interrupt(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][10] = 1
    Keys = ["R2", "Y"]
    return new_command, Keys

def Stop_Interrupt(last_command: Dict):
    new_command = deepcopy(last_command)
    new_command["Commands"][10] = 0
    Keys = ["R2", "A"]
    return new_command, Keys

def Construct_Commands():
    l = []
    keys_list = []
    # Gait Switch
    commands = deepcopy(Default_Standing)
    l.append(deepcopy(commands))
    commands, keys = Switch_Walking(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)

    commands, keys = Switch_Hopping(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)

    # Standing Commands:
    commands, keys = Switch_Standing(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)

    commands["Interval"] = 0.1
    for _ in range(7):
        commands, keys = Decrease_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1.5

    for _ in range(7):
        commands, keys = Increase_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    l[-1]["Interval"]=1.5 

    commands["Interval"] = 0.05
    for _ in range(14):
        commands, keys = Turning_Down_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    l[-1]["Interval"] =1

    for _ in range(14):
        commands, keys = Turning_Up_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    l[-1]["Interval"] =1.5

    commands["Interval"] = 0.05
    for _ in range(10):
        commands, keys = Turning_Left_Waist(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"] =1

    for _ in range(10):
        commands, keys = Turning_Right_Waist(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    l[-1]["Interval"] =1

    for _ in range(10):
        commands, keys = Turning_Right_Waist(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    l[-1]["Interval"] =1

    for _ in range(10):
        commands, keys = Turning_Left_Waist(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    
    l[-1]["Interval"] =1

    # Walking Commands:
    commands, keys = Switch_Walking(commands)
    commands["Commands"][3] = 1.5
    l.append(deepcopy(commands))
    keys_list.append(keys)

    # # Foot Height
    # commands["Interval"] = 0.4
    # for _ in range(8):
    #     commands, keys = Raise_Swing_Height(commands)
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)
    # l[-1]["Interval"] = 1.5

    # for _ in range(8):
    #     commands, keys = Put_down_Swing_Height(commands)
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)
    # l[-1]["Interval"] = 1.5

    # Freq
    commands["Interval"] = 0.5
    for _ in range(5):
        commands, keys = Increase_Gait_Frequency(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"] = 1

    for _ in range(5):
        commands, keys = Decrease_Gait_Frequency(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    # Walking Height
    commands["Interval"] = 0.2
    for _ in range(7):
        commands, keys = Decrease_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1.5

    for _ in range(7):
        commands, keys = Increase_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1.5
    
    # # Walking Pitch.
    # commands["Interval"] = 0.2
    # for _ in range(14):
    #     commands, keys = Turning_Down_Pitch(commands)
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)

    # l[-1]["Interval"] =1

    # for _ in range(14):
    #     commands, keys = Turning_Up_Pitch(commands)
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)

    # l[-1]["Interval"] =1.5


    commands["Interval"] = 0.8
    # Hopping
    commands, keys = Switch_Hopping(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)

    # Hopping Freq
    commands["Interval"] = 0.3
    for _ in range(5):
        commands, keys = Increase_Gait_Frequency(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"] = 1

    for _ in range(5):
        commands, keys = Decrease_Gait_Frequency(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    
    l[-1]["Interval"] = 1

    # Hopping Height
    commands["Interval"] = 0.2
    for _ in range(7):
        commands, keys = Decrease_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1

    for _ in range(7):
        commands, keys = Increase_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1

    # Standing
    commands, keys = Switch_Standing(commands)
    commands["Interval"] = 2
    l.append(deepcopy(commands))
    keys_list.append(keys)

    # Standing Pitch Down
    commands["Interval"] = 0.2
    for _ in range(12):
        commands, keys = Turning_Down_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    # l[-1]["Interval"] =10
    # Standing Pitch + Height Down
    commands["Interval"] = 0.25
    for _ in range(7):
        commands, keys = Decrease_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=2

    # Back Pitch
    for _ in range(12):
        commands, keys = Turning_Up_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    # Back Height
    for _ in range(7):
        commands, keys = Increase_Body_Height(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)

    l[-1]["Interval"] = 3
    
    # 磕头3次
    commands["Interval"] = 0.02
    for _ in range(14):
        commands, keys = Turning_Down_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=0.5
    # Back Pitch
    for _ in range(14):
        commands, keys = Turning_Up_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1

    commands["Interval"] = 0.02
    for _ in range(14):
        commands, keys = Turning_Down_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=0.5
    # Back Pitch
    for _ in range(14):
        commands, keys = Turning_Up_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=1

    commands["Interval"] = 0.02
    for _ in range(14):
        commands, keys = Turning_Down_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=0.5
    # Back Pitch
    for _ in range(14):
        commands, keys = Turning_Up_Pitch(commands)
        l.append(deepcopy(commands))
        keys_list.append(keys)
    l[-1]["Interval"]=30


    return l, keys_list
    
# print(Stage_Commands, Key_Lists)
# s = 0
# for cmd in Stage_Commands:
#     s+=cmd["Interval"]
# print(s)

def Construct_Interrupt_Commands():
    l = []
    keys_list = []
    # Gait Switch
    commands = deepcopy(Default_Standing)
    l.append(deepcopy(commands))
    # commands, keys = Switch_Walking(commands)
    # l.append(deepcopy(commands))
    # keys_list.append(keys)
    # commands, keys = Increase_Ang_z(commands)
    # commands, keys = Increase_Ang_z(commands)
    # commands, keys = Increase_Ang_z(commands)
    # for _ in range(10):
    #     commands, keys = Turning_Left_Waist(commands)
    #     commands["Interval"] = 0.1
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)

    # l.append(deepcopy(commands))
    # keys_list.append(keys)
    commands, keys = Start_Interrupt(commands) 
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 2.2
    # keys_list.append(keys)
    commands, keys = Switch_Hopping(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    # commands, keys = Increase_Lin_x(commands)
    # commands, keys = Increase_Lin_x(commands)
    # commands, keys = Increase_Lin_x(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = 2.0
    l[-1]["Interval"] = 0.35
    # keys_list.append(keys)
    commands, keys = Increase_Body_Height(commands)
    commands, keys = Increase_Body_Height(commands)
    commands, keys = Increase_Body_Height(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = 2.0
    l[-1]["Interval"] = 0.35
    commands, keys = Switch_Standing(commands)
    # l.append(deepcopy(commands))
    # l[-1]["Interval"] = 0.4

    # commands, keys = Switch_Walking(commands)
    for _ in range(10):
        commands, keys = Turning_Left_Waist(commands)
        l.append(deepcopy(commands))
        l[-1]["Interval"] = 0.07

    for _ in range(10):
        commands, keys = Turning_Right_Waist(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 0.7

    for _ in range(10):
        commands, keys = Turning_Right_Waist(commands)
        l.append(deepcopy(commands))
        l[-1]["Interval"] = 0.07
    commands, keys = Stop_Interrupt(commands)
    for _ in range(10):
        commands, keys = Turning_Left_Waist(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 0.5

    commands, keys = Switch_Hopping(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = -0.6
    l[-1]["Interval"] = 0.8
    # keys_list.append(keys)
    commands, keys = Switch_Standing(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    for _ in range(6):
        commands, keys = Turning_Down_Pitch(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 1.5
    commands, keys = Start_Interrupt(commands) 
    for _ in range(6):
        commands, keys = Turning_Up_Pitch(commands)
        commands, keys = Increase_Body_Height(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 0.5

    commands, keys = Switch_Hopping(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    # commands, keys = Increase_Lin_x(commands)
    # commands, keys = Increase_Lin_x(commands)
    # commands, keys = Increase_Lin_x(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = 2.0
    l[-1]["Interval"] = 0.35
    # keys_list.append(keys)
    commands, keys = Increase_Body_Height(commands)
    commands, keys = Increase_Body_Height(commands)
    commands, keys = Increase_Body_Height(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = 2.0
    l[-1]["Interval"] = 0.35
    commands, keys = Switch_Standing(commands)
    l.append(deepcopy(commands))

    l[-1]["Commands"][0] = 0.6
    commands, keys = Switch_Hopping(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    commands, keys = Decrease_Body_Height(commands)
    # commands, keys = Increase_Lin_x(commands)
    # commands, keys = Increase_Lin_x(commands)
    # commands, keys = Increase_Lin_x(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = 2.0
    l[-1]["Interval"] = 0.35
    # keys_list.append(keys)
    commands, keys = Increase_Body_Height(commands)
    commands, keys = Increase_Body_Height(commands)
    commands, keys = Increase_Body_Height(commands)
    l.append(deepcopy(commands))
    l[-1]["Commands"][0] = 2.0
    l[-1]["Interval"] = 0.35

    # commands, keys = Switch_Standing(commands)
    # l.append(deepcopy(commands))

    # l[-1]["Commands"][0] = 0.6

    # l.append(deepcopy(commands))
    # l[-1]["Interval"] = 1.1
    commands, keys = Switch_Standing(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 1.1
    commands, keys = Stop_Interrupt(commands)
    # commands, keys = Switch_Standing(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 5   

    return l, keys_list
Stage_Commands, Key_Lists = Construct_Interrupt_Commands()  # BOX

def Construct_Jump_Commands():
    l = []
    keys_list = []
    # Gait Switch
    commands = deepcopy(Default_Standing)
    l.append(deepcopy(commands))
    commands, _ = Switch_Jumping_Leftair(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 5
    commands, _ = Switch_Jumping_Rightair(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 1.08
    commands, _ = Switch_Jumping_Leftair(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 1.08 + 0.1
    commands, _ = Switch_Jumping_Rightair(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 0.88 - 0.1
    commands, _ = Switch_Jumping_Leftair(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 0.88
    commands, _ = Switch_Jumping_Rightair(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 0.9

    # commands, keys = Switch_Jumping_Leftair(commands) 
    # l.append(deepcopy(commands))
    # l[-1]["Interval"] = 10
    # keys_list.append(keys)
    # commands, keys = Switch_Jumping_Rightair(commands) 
    # l.append(deepcopy(commands))
    # l[-1]["Interval"] = 10
    # keys_list.append(keys)

    commands, keys = Switch_Standing(commands)
    l.append(deepcopy(commands))
    l[-1]["Interval"] = 10
    keys_list.append(keys)

    return l, keys_list



Stage_Commands, Key_Lists = Construct_Jump_Commands()

def Construct_Debug_Commands():
    l = []
    keys_list = []
    commands = deepcopy(Default_Standing)
    commands, keys = Start_Interrupt(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)
    l[-1]["Interval"] = 5
    # commands, keys = Switch_Walking(commands)
    # for _ in range(10):
    #     commands, keys = Increase_Lin_y(commands)
    #     commands, keys = Increase_Lin_y(commands)
    #     commands, keys = Increase_Lin_y(commands)
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)
    #     l[-1]["Interval"] = 4
    #     commands, keys = Decrease_Lin_y(commands)
    #     commands, keys = Decrease_Lin_y(commands)
    #     commands, keys = Decrease_Lin_y(commands)
    #     commands, keys = Decrease_Lin_y(commands)
    #     commands, keys = Decrease_Lin_y(commands)
    #     commands, keys = Decrease_Lin_y(commands)
    #     l.append(deepcopy(commands))
    #     keys_list.append(keys)
    #     l[-1]["Interval"] = 4
    #     commands, keys = Increase_Lin_y(commands)
    #     commands, keys = Increase_Lin_y(commands)
    #     commands, keys = Increase_Lin_y(commands)
    commands, keys = Switch_Standing(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)
    l[-1]["Interval"] = 300
    commands, keys = Stop_Interrupt(commands)
    l.append(deepcopy(commands))
    keys_list.append(keys)
    return l, keys_list

# Stage_Commands, Key_Lists = Construct_Debug_Commands()

def Construct_Interrupt_Commands_from_buffer(path = '../trajectories/noise_buffer_30update_0.1rad.pt', interval=30*0.02):
    import torch
    trajs = torch.load(path, map_location='cpu').cpu().numpy().astype(np.float32)
    ls = []
    default = np.zeros(8).astype(np.float32)
    print(len(trajs))
    for i in range(200):
        for j in range(len(trajs[i])):
            target = trajs[i,j]
            for k in range(60):
                rate = k/60
                current_cmd = JointPosInterpolation(default, target, rate)
                ls.append({
                    "Commands": current_cmd,
                    "Interval": 0.02
                })
            
            default = target
    target = np.zeros(8).astype(np.float32)
    for k in range(60):
        rate = k/60
        current_cmd = JointPosInterpolation(default, target, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": 0.02
        })
    return ls
def Construct_Interrupt_Commands_from_collecttraj(path = COLLECT_TRAJ_PATH):
    # Load all the subfoler in the path
    import os
    ls = []
    for folder in os.listdir(path):
        # Load the hdf5 data file
        # print(folder)
        np_name = os.path.join(path, folder, "traj_arm.npy")
        if not os.path.exists(np_name):
            file_name = os.path.join(path, folder, "data.hdf5")
            import h5py
            with h5py.File(file_name, 'r') as f:
                # import pdb; pdb.set_trace()
                traj_arm = np.concatenate((f["cmd_pos"][:, 2:6], f["cmd_pos"][:, 7:11]), axis=1)
                np.save(os.path.join(path, folder, "traj_arm.npy"), traj_arm)
                # print(traj_arm.shape)
                # print(f.keys())
                for i in range(traj_arm.shape[0]):
                    target = traj_arm[i]
                    ls.append({
                        "Commands": target*4,
                        "Interval": 0.02
                    })
        else:
            # print("AAA")
            traj_arm = np.load(np_name)
            for i in range(traj_arm.shape[0]):
                target = traj_arm[i]
                ls.append({
                    "Commands": target*4,
                    "Interval": 0.02
                })
        for i in range(30):
            cmd = JointPosInterpolation(target*4, np.zeros(8).astype(np.float32), i/30)
            ls.append({
                "Commands": cmd,
                "Interval": 0.02
            })
    return ls
                
def Construct_Interrupt_Commands_for_box(path = COLLECT_TRAJ_PATH):
    # Load all the subfoler in the path
    import os
    ls = []
    folder = "20250105-161208"
    np_name = os.path.join(path, folder, "traj_arm.npy")
    if not os.path.exists(np_name):
        file_name = os.path.join(path, folder, "data.hdf5")
        import h5py
        with h5py.File(file_name, 'r') as f:
            # import pdb; pdb.set_trace()
            traj_arm = np.concatenate((f["cmd_pos"][:, 2:6], f["cmd_pos"][:, 7:11]), axis=1)
            np.save(os.path.join(path, folder, "traj_arm.npy"), traj_arm)
            # print(traj_arm.shape)
            # print(f.keys())
            for i in range(traj_arm.shape[0]):
                target = traj_arm[i]
                ls.append({
                    "Commands": target*4.2,
                    "Interval": 0.02
                })
    else:
        traj_arm = np.load(np_name)
        for i in range(traj_arm.shape[0]):
            target = traj_arm[i]
            # if i<150:
            #     print(target[3], target[7])
            # if target[7]>0:
            #     target[7] *=2
           
            ls.append({
                "Commands": target*4.2,
                "Interval": 0.02
            })
            if 40<=i<=50 or 340<=i<=350:
                # ls[-1]["Interval"] = 0.
                ls[-1]["Commands"][7] *= 2.2
                ls[-1]["Commands"][4] *= 1.8
            if 92 <=i <= 104 or 402<=i<=412:
                ls[-1]["Commands"][3] *= 2.2
                ls[-1]["Commands"][0] *= 1.8
    for i in range(30):
        cmd = JointPosInterpolation(target*4.2, np.zeros(8).astype(np.float32), i/30)
        ls.append({
            "Commands": cmd,
            "Interval": 0.02
        })
    return ls         

def Construct_Interrupt_Commands_for_amass(path = "/home/dongwentao/Verstile-Locomotion/data/phc_data"):
    # Possible Traj: 203 215 223 235
    import os
    ls = []
    last_idx =  0 
    for i in range(203, 400): #14400
        file_name = os.path.join(path, "amass_phc_filtered_{}.npy".format(i))
        trajs = np.load(file_name)
        for j in range(trajs.shape[0]):
            target = trajs[j]
            ls.append({
                "Commands": target*4,
                "Interval": 0.02
            })
        for j in range(30):
            cmd = JointPosInterpolation(target*4, np.zeros(8).astype(np.float32), i/30)
            ls.append({
                "Commands": cmd,
                "Interval": 0.02
            })
        if (last_idx < 14400 and len(ls) >=14400):
            print("IDX:", i)
        last_idx = len(ls)
    return ls
def JointPosInterpolation(initPos, targetPos, rate):
    rate = min(rate, 1)
    Pos = initPos * (1 - rate) + targetPos * rate
    return Pos

def Construct_Interrupt_Hello(raise_time=2, waving_time= 1, waving_num=10, dt = 0.05):
    # Only waving Left hands:
    commands = np.zeros(8).astype(np.float32)
    ls = []
    # Raise your left hand:
    for t in np.arange(dt, raise_time+dt, dt):
        rate = t / raise_time
        commands[1] = JointPosInterpolation(0, 8.0, rate)
        commands[2] = JointPosInterpolation(0, 6.0, rate)
        ls.append({
            "Commands": deepcopy(commands),
            "Interval": dt
        })

    for _ in range(waving_num):
        for t in np.arange(0, 1, dt/waving_time):
            if t < 0.25:
                commands[3] = -3 * (4 * t) 
            elif 0.25 <= t < 0.5:
                commands[3] = -3 * (1 - 4 * (t-0.25))
            elif 0.5<= t < 0.75:
                commands[3] = 4 * (4 * (t-0.5))
            else:
                commands[3] = 4 * (1- 4*(t-0.75))
            ls.append({
                "Commands": deepcopy(commands),
                "Interval": dt
            })

    # Put down your left hand:
    for t in np.arange(dt, raise_time+dt, dt):
        rate = t / raise_time
        commands[1] = JointPosInterpolation(8.0, 0, rate)
        commands[2] = JointPosInterpolation(6.0, 0, rate)
        ls.append({
            "Commands": deepcopy(commands),
            "Interval": dt
        })
    ls[-1]["Interval"] = 2
    return ls

def Construct_Interrupt_Pick():

    init_default_pos = np.zeros(8).astype(np.float32)
    target_1 = np.array([-0.55, 0.4, 0.2, 0.7, -0.55, -0.4, -0.2, 0.7]).astype(np.float32)
    target_2 = np.array([-0.5, -0.05, 0.2, 0.7, -0.5, -0.05, -0.2, 0.7]).astype(np.float32)
    target_3 = np.array([-1.6, 0.1, 0.2, 1.4, -1.6, -0.1, -0.2, 1.4]).astype(np.float32)
    seq_time = 2
    dt = 0.05
    ls = []
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(init_default_pos, target_1 * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(target_1*4 , target_2 * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(target_2*4 , target_3 * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })
    ls[-1]["Interval"] = 30 # Hold 10s
    return ls

def Construct_Interrupt_Pick_second_part():
    
    init_default_pos = np.zeros(8).astype(np.float32)
    target_1 = np.array([-0.55, 0.4, 0.2, 0.7, -0.55, -0.4, -0.2, 0.7]).astype(np.float32)
    target_2 = np.array([-0.5, -0.05, 0.3, 0.7, -0.5, 0.05, -0.3, 0.7]).astype(np.float32)
    target_3 = np.array([-0.5, -0.05, -0.04, 0.3, -0.5, 0.05, 0.04, 0.3]).astype(np.float32)
    target_4 = np.array([-1.6, 0.2, 0.3, 1.4, -1.6, -0.2, -0.3, 1.4]).astype(np.float32)
    seq_time = 2
    dt = 0.05
    ls = []
    
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(target_4*4 , target_3 * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(target_3*4 , target_2 * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(target_2*4 , target_1 * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })

    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        commands = JointPosInterpolation(target_1*4 , init_default_pos * 4, rate)
        ls.append({
            "Commands": commands,
            "Interval": dt
        })
    ls[-1]["Interval"] = 5
    return ls

# def Construct_Interrupt_Pick():

#     init_default_pos = np.zeros(8).astype(np.float32)
#     target_1 = np.array([-0.55, 0.4, 0.2, 0.7, -0.55, -0.4, -0.2, 0.7]).astype(np.float32)
#     target_2 = np.array([-0.5, -0.05, 0.3, 0.7, -0.5, 0.05, -0.3, 0.7]).astype(np.float32)
#     target_3 = np.array([-0.5, -0.05, -0.04, 0.3, -0.5, 0.05, 0.04, 0.3]).astype(np.float32)
#     target_4 = np.array([-1.6, 0.2, 0.3, 1.4, -1.6, -0.2, -0.3, 1.4]).astype(np.float32)
#     # -1.6, 0.1, 0.2, 1.4, -1.6, -0.1, -0.2, 1.4
#     seq_time = 2
#     dt = 0.05
#     ls = []
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(init_default_pos, target_1 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_1*4 , target_2 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_2*4 , target_3 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_3*4 , target_4 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     ls[-1]["Interval"] =  30 # Hold 30s
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_4*4 , target_3 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_3*4 , target_2 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_2*4 , target_1 * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })

#     for t in np.arange(dt, seq_time + dt, dt):
#         rate = t/seq_time
#         commands = JointPosInterpolation(target_1*4 , init_default_pos * 4, rate)
#         ls.append({
#             "Commands": commands,
#             "Interval": dt
#         })
#     ls[-1]["Interval"] = 5
#     return ls


def Construct_Interrupt_Pick_2():
    seq_time = 2
    dt = 0.05
    ls = []
    default = np.array([0,0,0,0,0,0,0,0]).astype(np.float32)
    target = np.array([-2.4, -0.6, 0, 2.4, -2.4, 0.6, 0, 2.4]).astype(np.float32)
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        current_cmd = JointPosInterpolation(default, target, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": dt
        })
    ls[-1]["Interval"] = 10
    for t in np.arange(dt, seq_time + dt, dt):
        rate = t/seq_time
        current_cmd = JointPosInterpolation(target, default, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": dt
        })
    ls[-1]["Interval"] = 10
    
    return ls

def Construct_Interrupt_Waving(waving_num=10):
    default = np.array([0,0,0,0,0,0,0,0]).astype(np.float32) * 4
    target_out = np.array([0, 1.57, 0, 1.57, 0, -1.57, 0, 1.57]).astype(np.float32) * 4
    target_mid = np.array([-1.2, 0.26, -1, 1.4, -0.8, -0.2, 1, 1.4]).astype(np.float32) * 4
    target_in = np.array([-1.2, -0.14, -1.1, 0.39, -0.8, 0.18, 1.3, 0.28]).astype(np.float32) * 4
    seq_time_in = 0.3
    seq_time_out = 0.8
    dt = 0.05
    ls = []
    for t in np.arange(dt, seq_time_out + dt, dt):
        rate = t/seq_time_out
        current_cmd = JointPosInterpolation(default, target_out, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": dt
        })
    for _ in range(waving_num):
        for t in np.arange(dt, seq_time_out + dt, dt):
            rate = t/seq_time_out
            current_cmd = JointPosInterpolation(target_out, target_mid, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
        for t in np.arange(dt, seq_time_in + dt, dt):
            rate = t/seq_time_in
            current_cmd = JointPosInterpolation(target_mid, target_in, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
        for t in np.arange(dt, seq_time_in + dt, dt):
            rate = t/seq_time_in
            current_cmd = JointPosInterpolation(target_in, target_mid, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
        for t in np.arange(dt, seq_time_out + dt, dt):
            rate = t/seq_time_out
            current_cmd = JointPosInterpolation(target_mid, target_out, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
    for t in np.arange(dt, seq_time_out + dt, dt):
        rate = t/seq_time_out
        current_cmd = JointPosInterpolation(target_out, default, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": dt
        })
    ls[-1]["Interval"] = 10
    return ls

def Construct_Interrupt_Commands_for_box(path = COLLECT_TRAJ_PATH):
    # Load all the subfoler in the path
    import os
    ls = []
    folder = "20250105-161208"
    np_name = os.path.join(path, folder, "traj_arm.npy")
    if not os.path.exists(np_name):
        file_name = os.path.join(path, folder, "data.hdf5")
        import h5py
        with h5py.File(file_name, 'r') as f:
            # import pdb; pdb.set_trace()
            traj_arm = np.concatenate((f["cmd_pos"][:, 2:6], f["cmd_pos"][:, 7:11]), axis=1)
            np.save(os.path.join(path, folder, "traj_arm.npy"), traj_arm)
            # print(traj_arm.shape)
            # print(f.keys())
            for i in range(traj_arm.shape[0]):
                target = traj_arm[i]
                ls.append({
                    "Commands": target*4,
                    "Interval": 0.02
                })
    else:
        traj_arm = np.load(np_name)
        for i in range(traj_arm.shape[0]):
            target = traj_arm[i]
            # if i<150:
            #     print(target[3], target[7])
            # if target[7]>0:
            #     target[7] *=2
           
            ls.append({
                "Commands": target*4,
                "Interval": 0.02
            })
            if 40<=i<=50 or 340<=i<=350:
                # ls[-1]["Interval"] = 0.
                ls[-1]["Commands"][7] *= 2.3
                ls[-1]["Commands"][4] *= 1.3
            if 92 <=i <= 104 or 402<=i<=412:
                ls[-1]["Commands"][3] *= 2.3
                ls[-1]["Commands"][0] *= 1.3
    for i in range(30):
        cmd = JointPosInterpolation(target*4, np.zeros(8).astype(np.float32), i/30)
        ls.append({
            "Commands": cmd,
            "Interval": 0.02
        })
    return ls       


def Construct_Interrupt_Waving(waving_num=10):
    default = np.array([0,0,0,0,0,0,0,0]).astype(np.float32) * 4
    target_out = np.array([0, 1.57, 0, 1.57, 0, -1.57, 0, 1.57]).astype(np.float32) * 4
    target_mid = np.array([-1.2, 0.26, -1, 1.4, -0.8, -0.2, 1, 1.4]).astype(np.float32) * 4
    target_in = np.array([-1.2, -0.14, -1.1, 0.39, -0.8, 0.18, 1.3, 0.28]).astype(np.float32) * 4
    seq_time_in = 0.3
    seq_time_out = 0.8
    dt = 0.05
    ls = []
    for t in np.arange(dt, seq_time_out + dt, dt):
        rate = t/seq_time_out
        current_cmd = JointPosInterpolation(default, target_out, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": dt
        })
    for _ in range(waving_num):
        for t in np.arange(dt, seq_time_out + dt, dt):
            rate = t/seq_time_out
            current_cmd = JointPosInterpolation(target_out, target_mid, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
        for t in np.arange(dt, seq_time_in + dt, dt):
            rate = t/seq_time_in
            current_cmd = JointPosInterpolation(target_mid, target_in, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
        for t in np.arange(dt, seq_time_in + dt, dt):
            rate = t/seq_time_in
            current_cmd = JointPosInterpolation(target_in, target_mid, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
        for t in np.arange(dt, seq_time_out + dt, dt):
            rate = t/seq_time_out
            current_cmd = JointPosInterpolation(target_mid, target_out, rate)
            ls.append({
                "Commands": current_cmd,
                "Interval": dt
            })
    for t in np.arange(dt, seq_time_out + dt, dt):
        rate = t/seq_time_out
        current_cmd = JointPosInterpolation(target_out, default, rate)
        ls.append({
            "Commands": current_cmd,
            "Interval": dt
        })
    ls[-1]["Interval"] = 10
    return ls

if INTERRUPT_MODE == "Noise":
    Interrupt_Commands = Construct_Interrupt_Commands_from_buffer()
elif INTERRUPT_MODE == "Hello":
    Interrupt_Commands = Construct_Interrupt_Hello(waving_time=1, waving_num=5)
# elif INTERRUPT_MODE == "Pick":
#     Interrupt_Commands = Construct_Interrupt_Pick()
elif INTERRUPT_MODE == "Pick":
    Interrupt_Commands = Construct_Interrupt_Pick()
    Interrupt_Commands2 = Construct_Interrupt_Pick_second_part()
elif INTERRUPT_MODE == "Pick2":
    Interrupt_Commands = Construct_Interrupt_Pick_2()
elif INTERRUPT_MODE == "Traj":
    Interrupt_Commands = Construct_Interrupt_Commands_from_collecttraj()
elif INTERRUPT_MODE == "Box":
    Interrupt_Commands = Construct_Interrupt_Commands_for_box()
elif INTERRUPT_MODE == "Amass":
    Interrupt_Commands = Construct_Interrupt_Commands_for_amass()