import importlib
import numpy as np
import torch
import random
try:
    from absl import flags
    FLAGS = flags.FLAGS
    try:
        FLAGS(["train_sc.py"])
    except:
        pass
except ImportError:
    FLAGS = None

from harl.envs.smac.smac_logger import SMACLogger
from harl.envs.smacv2.smacv2_logger import SMACv2Logger
from harl.envs.mamujoco.mamujoco_logger import MAMuJoCoLogger
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
from harl.envs.gym.gym_logger import GYMLogger
from harl.envs.football.football_logger import FootballLogger
from harl.envs.dexhands.dexhands_logger import DexHandsLogger
from harl.envs.lag.lag_logger import LAGLogger
from harl.envs.v2x.v2x_logger import V2XLogger

LOGGER_REGISTRY = {
    "smac": SMACLogger,
    "mamujoco": MAMuJoCoLogger,
    "pettingzoo_mpe": PettingZooMPELogger,
    "gym": GYMLogger,
    "football": FootballLogger,
    "dexhands": DexHandsLogger,
    "smacv2": SMACv2Logger,
    "lag": LAGLogger,
    "v2x": V2XLogger,
}


def get_task_name(env_name, env_args):
    if env_name == "smac":
        task_name = env_args["map_name"]
    elif env_name == "smacv2":
        task_name = env_args["map_name"]
    elif env_name == "mamujoco":
        task_name = env_args["scenario"]
    elif env_name == "pettingzoo_mpe":
        task_name = env_args["scenario"]
    elif env_name == "football":
        task_name = env_args["env_name"]
    elif env_name == "dexhands":
        task_name = env_args["task"]["name"]
    elif env_name == "gym":
        task_name = env_args["scenario"]
    elif env_name == "lag":
        task_name = env_args["task"]["name"]
    elif env_name == "v2x":
        task_name = "v2x_task"  # 为v2x环境指定一个任务名
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    return task_name


def init_env(env_name, env_args, seed):
    """Initialize environment."""
    # --- 【核心诊断代码】 ---
    # 我们在这里打印出函数接收到的env_name，看看它到底是什么。
    print(f"--- [DEBUG] Entering init_env. Received env_name = '{env_name}' ---")
    # -------------------------

    if env_name == "smac":
        from harl.envs.smac.StarCraft2_Env import StarCraft2Env

        env = StarCraft2Env(env_args)
    elif env_name == "smacv2":
        from harl.envs.smacv2.smacv2_env import SMACv2

        env = SMACv2(env_args)
    elif env_name == "mamujoco":
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti

        env = MujocoMulti(env_args=env_args)
    elif env_name == "pettingzoo_mpe":
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import (
            PettingZooMPEEnv,
        )

        env = PettingZooMPEEnv(env_args)
    elif env_name == "football":
        from harl.envs.football.football_env import FootballEnv

        env = FootballEnv(env_args)
    elif env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexterousHandsEnv

        env = DexterousHandsEnv(env_args)
    elif env_name == "gym":
        from harl.envs.gym.gym_env import GymEnv

        env = GymEnv(env_args)
    elif env_name == "lag":
        from harl.envs.lag.lag_env import LagEnv

        env = LagEnv(env_args)
    # --- 【新增】为我们的v2x环境添加入口 ---
    elif env_name == "v2x":
        from harl.envs.v2x.v2x_env import V2XTaskOffloadingEnv
        print("[DEBUG] Matched 'v2x'. Creating V2XTaskOffloadingEnv...")
        env = V2XTaskOffloadingEnv(env_args)
    # ------------------------------------
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    env.seed(seed)
    return env


def make_train_env(env_name, seed, n_threads, env_args):
    """Make parallel environments for training."""
    def get_env_fn(rank):
        def init_env_():
            env = init_env(env_name, env_args, seed + rank * 1000)
            return env

        return init_env_

    if n_threads == 1:
        # Use single-process environment for n_threads=1 to avoid BrokenPipeError
        from harl.envs.env_wrappers import ShareDummyVecEnv
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        # Use multi-process environment for n_threads>1
        from harl.envs.env_wrappers import ShareSubprocVecEnv
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make parallel environments for evaluation."""
    def get_env_fn(rank):
        def init_env_():
            env = init_env(env_name, env_args, seed + rank * 1000)
            return env

        return init_env_

    if n_threads == 1:
        # Use single-process environment for n_threads=1 to avoid BrokenPipeError
        from harl.envs.env_wrappers import ShareDummyVecEnv
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    else:
        # Use multi-process environment for n_threads>1
        from harl.envs.env_wrappers import ShareSubprocVecEnv
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make environment for rendering."""
    # As for rendering, we only need to use one environment.
    # So we set n_threads to 1.
    manual_render = False
    manual_expand_dims = False
    manual_delay = False
    env_num = 1
    if env_name == "smac" or env_name == "smacv2":
        env = init_env(env_name, env_args, seed)
    elif env_name == "mamujoco":
        env = init_env(env_name, env_args, seed)
    elif env_name == "pettingzoo_mpe":
        env = init_env(env_name, env_args, seed)
    elif env_name == "football":
        env = init_env(env_name, env_args, seed)
    elif env_name == "dexhands":
        env = init_env(env_name, env_args, seed)
        manual_render = True
        manual_expand_dims = True
        manual_delay = True
        env_num = env_args["env"]["num_envs"]
    elif env_name == "gym":
        env = init_env(env_name, env_args, seed)
    elif env_name == "lag":
        env = init_env(env_name, env_args, seed)
    elif env_name == "v2x": # 新增v2x的渲染分支
        env = init_env(env_name, env_args, seed)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return env, manual_render, manual_expand_dims, manual_delay, env_num


def get_num_agents(env_name, env_args, envs):
    if env_name == "smac" or env_name == "smacv2":
        num_agents = envs.n_agents
    elif env_name == "mamujoco":
        num_agents = envs.n_agents
    elif env_name == "pettingzoo_mpe":
        num_agents = envs.n_agents
    elif env_name == "football":
        num_agents = envs.n_agents
    elif env_name == "dexhands":
        num_agents = envs.num_agents
    elif env_name == "gym":
        num_agents = envs.n_agents
    elif env_name == "lag":
        num_agents = envs.n_agents
    elif env_name == "v2x": # 新增v2x的智能体数量获取
        num_agents = envs.num_agents
    else:
        raise ValueError(f"Unsupported environment: {env_name}")
    return num_agents


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space."""
    if hasattr(obs_space, 'shape'):
        return obs_space.shape
    else:
        return obs_space.n


def get_shape_from_act_space(act_space):
    """Get shape from action space."""
    if act_space.__class__.__name__ == "Discrete":
        return act_space.n
    elif act_space.__class__.__name__ == "Box":
        return act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        return act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiDiscrete":
        return act_space.shape[0]


def check(input):
    """Check if the input is a tensor, if not, convert it to a tensor."""
    if input is None:
        return None
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        # 尝试转换为numpy数组再转换为tensor
        try:
            return torch.from_numpy(np.array(input))
        except Exception:
            return torch.tensor(input)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
