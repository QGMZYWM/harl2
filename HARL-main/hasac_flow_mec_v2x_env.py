import numpy as np
import gym
from gym import spaces
import random

# 这是一个简化的MEC-V2X环境模拟，用于与HARL框架集成
# 它旨在提供与您项目描述一致的API接口

class MECVehicularEnvironment(gym.Env):
    """
    简化的多智能体MEC-V2X环境。
    此版本修正了API以兼容HARL框架。
    """
    def __init__(self, num_vehicles=5, num_rsus=2, map_size=1000.0, max_history_length=50):
        super(MECVehicularEnvironment, self).__init__()

        self.num_vehicles = num_vehicles
        self.num_agents = num_vehicles
        self.num_rsus = num_rsus
        self.map_size = map_size
        self.max_history_length = max_history_length

        # 状态维度：[x, y, vx, vy, task_size, task_deadline]
        self.obs_dim_single = 6
        # 动作维度：[offload_target, power]
        self.action_dim_single = 2

        # FIX 1: 定义与HARL兼容的观察和动作空间 (使用gym.spaces.Dict)
        self.observation_space = spaces.Dict({
            f'vehicle_{i}': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim_single,), dtype=np.float32)
            for i in range(self.num_vehicles)
        })
        
        # FIX: 将action_space从Dict改为Tuple，以符合HARL框架要求
        action_spaces = [spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim_single,), dtype=np.float32) 
                        for _ in range(self.num_vehicles)]
        self.action_space = spaces.Tuple(action_spaces)
        
        # 为中心化评论家定义共享观察空间
        self.share_observation_space = spaces.Dict({
            f'vehicle_{i}': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.obs_dim_single * self.num_vehicles,), 
                dtype=np.float32
            )
            for i in range(self.num_vehicles)
        })

        self.vehicles = {}
        self.rsus = {}
        self.current_step = 0
        self.max_steps = 200

    def reset(self):
        """
        重置环境状态。
        FIX 2: 返回一个符合`observation_space`结构的字典。
        """
        self.current_step = 0
        for i in range(self.num_vehicles):
            self.vehicles[f'vehicle_{i}'] = {
                'pos': np.random.rand(2) * self.map_size,
                'vel': np.random.randn(2) * 10,
                'task_size': random.uniform(100, 1000),
                'task_deadline': random.uniform(5, 20)
            }
        
        for i in range(self.num_rsus):
            self.rsus[f'rsu_{i}'] = {
                'pos': np.random.rand(2) * self.map_size,
                'load': 0
            }
        
        return self._get_obs()

    def step(self, actions):
        """
        执行一个时间步。
        FIX 3: 返回符合HARL API的(obs, rewards, dones, infos)元组。
        """
        self.current_step += 1
        
        # 模拟车辆移动和任务处理
        for i in range(self.num_vehicles):
            vehicle_key = f'vehicle_{i}'
            self.vehicles[vehicle_key]['pos'] += self.vehicles[vehicle_key]['vel'] * 0.1
            # 边界处理
            self.vehicles[vehicle_key]['pos'] = np.clip(self.vehicles[vehicle_key]['pos'], 0, self.map_size)

        # 计算奖励、完成状态等
        rewards = {f'vehicle_{i}': random.uniform(-1, 1) for i in range(self.num_vehicles)}
        
        # 完成状态
        done = self.current_step >= self.max_steps
        dones = {f'vehicle_{i}': done for i in range(self.num_vehicles)}
        dones['__all__'] = done

        # 信息字典
        infos = {f'vehicle_{i}': {} for i in range(self.num_vehicles)}

        return self._get_obs(), rewards, dones, infos

    def _get_obs(self):
        """
        获取所有智能体的观察。
        返回一个字典，键为 'vehicle_i'。
        """
        obs_dict = {}
        for i in range(self.num_vehicles):
            v = self.vehicles[f'vehicle_{i}']
            obs = np.concatenate([
                v['pos'], v['vel'], [v['task_size']], [v['task_deadline']]
            ])
            obs_dict[f'vehicle_{i}'] = obs.astype(np.float32)
        return obs_dict

    def render(self, mode='human'):
        pass

    def close(self):
        pass

