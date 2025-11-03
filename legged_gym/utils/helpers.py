import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def export_policy_as_jit(actor_critic, path, example_obs):
    from rsl_rl.modules.net_model import MlpAdaptModel
    from rsl_rl.modules.actor_critic import ActorCritic
    if hasattr(actor_critic, "student_model_name") and 'Rnn' in actor_critic.student_model_name:
        exporter = StudentPolicyExporter(actor_critic)
        exporter.export(path, example_obs)
    elif isinstance(actor_critic, ActorCritic) and isinstance(actor_critic.actor, MlpAdaptModel):
        exporter = AsymetricPolicyExporter(actor_critic)
        exporter.export(path, example_obs)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "h1int", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": True,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--sim_joystick", "action": "store_true", "default":False, "help": "Sample commands from sim joystick"},
        {"name": "--no_clock", "type": bool, "default": False, "help": "No clock input."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args




class StudentPolicyExporter(torch.nn.Module):
    def __init__(self, student_policy):
        super().__init__()
        for para in student_policy.parameters():
            para.requires_grad = False
        
        self.latent_head = copy.deepcopy(student_policy.actor.latent_head)
        self.teacher_low_level = copy.deepcopy(student_policy.teacher_low_level)
        self.memory = copy.deepcopy(student_policy.actor.memory_encoder.rnn)

        self.proprioception_dim = student_policy.proprioception_dim
        # self.low_level_obs_dim = student_policy.low_level_obs_dim
        
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size, dtype=torch.float32))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size, dtype=torch.float32))

    def forward(self, x):
        low_level_obs = x
        if low_level_obs.dim() == 3:
            low_level_obs = low_level_obs[:,-1,:]

        memory, (h,c) = self.memory(low_level_obs[..., :self.proprioception_dim].unsqueeze(0), (self.hidden_state, self.cell_state))
        student_latent = self.latent_head(memory.squeeze(0))

        low_state = torch.cat((low_level_obs, student_latent), dim=-1)
        student_act = self.teacher_low_level(low_state)

        self.hidden_state[:] = h
        self.cell_state[:] = c

        return student_act, student_latent

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path, test_input):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'trace_jit.pt')
        self.to('cpu')
        self.eval()
        test = self(test_input)
        traced_policy = torch.jit.trace(self, test_input)
        traced_policy.save(path)
        # traced_script_module = torch.jit.script(self)
        # traced_script_module.save(path)

class AsymetricPolicyExporter(torch.nn.Module):
    # The Input Policy should belongs to class Teacher
    # And its actor should belongs to class MlpAdaptModel
    # Treat the obs_buffer as part of the module. 
    def __init__(self, AsymPolicy, num_envs=1):
        super().__init__()
        for para in AsymPolicy.parameters():
            para.requires_grad = False
        self.mem_encoder = copy.deepcopy(AsymPolicy.actor.mem_encoder)
        self.state_estimator = copy.deepcopy(AsymPolicy.actor.state_estimator)
        self.low_level_net = copy.deepcopy(AsymPolicy.actor.low_level_net)
        self.num_envs = num_envs

        self.proprioception_dim = AsymPolicy.actor.proprioception_dim
        self.include_history_steps = AsymPolicy.actor.max_length
        self.cmd_dim = AsymPolicy.actor.cmd_dim
        
        # Here we assume the obs always have num_envs = 1 for jit inferance.
        self.register_buffer(f'pro_obs_seq', torch.zeros(num_envs, self.include_history_steps, self.proprioception_dim, dtype=torch.float32)) 
        self.pro_obs_seq.requires_grad = False

        # self.low_level_obs_dim = student_policy.low_level_obs_dim
        # self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size, dtype=torch.float32))
        # self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size, dtype=torch.float32))

    def forward(self, x):
        low_level_obs = x
        if low_level_obs.dim() == 3:
            low_level_obs = low_level_obs[:,-1,:] # num_envs x proprioception_dim

        # Update history buffer.
        self.update_memory(low_level_obs)
        # self.pro_obs_seq[:, :-1, :] = self.pro_obs_seq[:, 1:, :].clone()
        # self.pro_obs_seq[:, -1, :] = low_level_obs[:, :self.proprioception_dim].clone()
        mem = self.mem_encoder(self.pro_obs_seq.view(1, -1))
        privileged_predict = self.state_estimator(mem)
        cmd = low_level_obs[..., self.proprioception_dim:self.proprioception_dim+self.cmd_dim]
        # act = self.low_level_net(torch.cat((x[..., -1, :self.proprioception_dim], mem, privileged_predict, cmd), dim=-1))
        # act = self.low_level_net(torch.cat((mem, cmd), dim=-1))
        jp_out = self.low_level_net(torch.cat((mem, privileged_predict, low_level_obs[..., :self.proprioception_dim], cmd), dim=-1))

        return jp_out
    
    @torch.jit.export
    def update_memory(self, low_level_obs):
        tmp = torch.concatenate((self.pro_obs_seq[:, 1:, :], low_level_obs[:, :self.proprioception_dim].unsqueeze(1)),dim=1)
        self.pro_obs_seq[:] = tmp


    @torch.jit.export
    def reset_memory(self):
        self.pro_obs_seq[:] = 0
        
 
    def export(self, path, test_input):
        os.makedirs(path, exist_ok=True)
        self.to('cpu')
        self.eval()
        traced_policy = torch.jit.trace(self, test_input)
        traced_policy.save(os.path.join(path, 'trace_jit.pt'))
        traced_policy = torch.jit.load(os.path.join(path, 'trace_jit.pt'), map_location=test_input.device)
        # print(traced_policy.pro_obs_seq)
        # traced_policy.reset_memory()
        # print(traced_policy.pro_obs_seq)
        # import pdb; pdb.set_trace()
        for _ in range(5):
            test_input = torch.rand_like(test_input).to(test_input.device).to(torch.float32)
            self(test_input)
            traced_policy(test_input)
        # print(self.pro_obs_seq - traced_policy.pro_obs_seq)
        # traced_policy.reset_memory()
        print(torch.sum(self.pro_obs_seq - traced_policy.pro_obs_seq))
        for _ in range(20):
            test_input = torch.rand_like(test_input).to(test_input.device).to(torch.float32)
            test = self(test_input)
            test_traced = traced_policy(test_input)
            print(test-test_traced)
            print(torch.sum(self.pro_obs_seq - traced_policy.pro_obs_seq))

