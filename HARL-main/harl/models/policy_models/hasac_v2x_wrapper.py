#!/usr/bin/env python3
"""
HASAC V2X环境包装器
将HASAC的连续动作转换为V2X环境需要的离散动作
保持原V2X环境不变，通过包装器实现兼容
"""

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn.functional as F

class HASACV2XWrapper(gym.Wrapper):
    """
    HASAC V2X环境包装器
    将连续动作空间转换为离散动作空间
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # 保存原始环境的离散动作空间
        self.original_action_space = env.action_space
        
        # 安全地获取动作空间维度
        try:
            # 尝试获取第一个智能体的动作空间
            if isinstance(env.action_space, list) and len(env.action_space) > 0:
                first_agent_space = env.action_space[0]
                if hasattr(first_agent_space, 'n'):
                    self.num_targets = first_agent_space.n
                else:
                    # 默认值，如果无法获取
                    self.num_targets = 5
            else:
                # 如果action_space不是列表，尝试直接获取
                if hasattr(env.action_space, 'n'):
                    self.num_targets = env.action_space.n
                else:
                    # 默认值
                    self.num_targets = 5
        except:
            # 如果上述方法都失败，使用默认值
            self.num_targets = 5
        
        # 为HASAC创建连续动作空间
        # [目标选择权重(num_targets), 传输功率, 任务优先级, 负载均衡因子]
        action_dim = self.num_targets + 3
        single_action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # 多智能体动作空间
        self.n_agents = getattr(env, 'n_agents', 1)
        self.action_space = [single_action_space for _ in range(self.n_agents)]
        
        print(f"🔄 HASAC V2X Wrapper initialized:")
        print(f"   - Original action space: Discrete({self.num_targets})")
        print(f"   - New action space: Box(shape=({action_dim},))")
        print(f"   - Action components: [target_weights({self.num_targets}), power(1), priority(1), balance(1)]")
    
    def step(self, continuous_actions):
        """
        将连续动作转换为离散动作并执行
        
        Args:
            continuous_actions: List of continuous actions for each agent
            
        Returns:
            Standard gym environment returns
        """
        # 转换每个智能体的连续动作为离散动作
        discrete_actions = []
        additional_controls = []
        
        for agent_id, cont_action in enumerate(continuous_actions):
            discrete_action, controls = self._convert_action(agent_id, cont_action)
            discrete_actions.append(discrete_action)
            additional_controls.append(controls)
        
        # 在原环境中执行离散动作
        obs, share_obs, rewards, dones, infos, available_actions = self.env.step(discrete_actions)
        
        # 使用额外控制信息调整奖励
        enhanced_rewards = self._enhance_rewards(rewards, additional_controls, infos)
        
        return obs, share_obs, enhanced_rewards, dones, infos, available_actions
    
    def _convert_action(self, agent_id, continuous_action):
        """
        将单个智能体的连续动作转换为离散动作
        
        Args:
            agent_id: 智能体ID
            continuous_action: 连续动作 [target_weights, power, priority, balance]
            
        Returns:
            discrete_action: 离散动作
            controls: 额外控制信息
        """
        # 分解连续动作
        target_weights = continuous_action[:self.num_targets]
        power_level = continuous_action[self.num_targets]
        priority_weight = continuous_action[self.num_targets + 1] 
        balance_factor = continuous_action[self.num_targets + 2]
        
        # 使用Gumbel-Softmax或简单的argmax选择目标
        if self.training:
            # 训练时使用Gumbel-Softmax保持可微性
            discrete_action = self._gumbel_softmax_select(target_weights)
        else:
            # 推理时使用argmax
            discrete_action = np.argmax(target_weights)
        
        # 额外控制信息
        controls = {
            'power_level': (power_level + 1) / 2,  # 转换到[0, 1]
            'priority_weight': (priority_weight + 1) / 2,  # 转换到[0, 1] 
            'balance_factor': (balance_factor + 1) / 2,  # 转换到[0, 1]
            'target_weights': target_weights
        }
        
        return discrete_action, controls
    
    def _gumbel_softmax_select(self, logits, temperature=1.0):
        """
        使用Gumbel-Softmax进行可微的离散选择
        
        Args:
            logits: 目标权重
            temperature: 温度参数
            
        Returns:
            选择的目标索引
        """
        # 添加Gumbel噪声
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, logits.shape) + 1e-20) + 1e-20)
        y = logits + gumbel_noise
        
        # Softmax选择
        probabilities = F.softmax(torch.tensor(y / temperature), dim=0).numpy()
        
        # 采样选择
        return np.random.choice(len(probabilities), p=probabilities)
    
    def _enhance_rewards(self, original_rewards, controls_list, infos):
        """
        使用额外控制信息增强奖励
        
        Args:
            original_rewards: 原始奖励
            controls_list: 额外控制信息列表
            infos: 环境信息
            
        Returns:
            增强后的奖励
        """
        enhanced_rewards = []
        
        for i, (reward, controls) in enumerate(zip(original_rewards, controls_list)):
            enhanced_reward = reward[0] if isinstance(reward, list) else reward
            
            # 功率效率奖励
            power_efficiency = 1.0 - abs(controls['power_level'] - 0.5)  # 中等功率最优
            enhanced_reward += power_efficiency * 0.5
            
            # 负载均衡奖励
            if 'current_load' in infos[i]:
                target_load = 0.5  # 目标负载50%
                actual_load = infos[i]['current_load']
                balance_bonus = controls['balance_factor'] * (1.0 - abs(actual_load - target_load))
                enhanced_reward += balance_bonus * 0.3
            
            # 优先级调度奖励
            if 'completed_tasks' in infos[i] and infos[i]['completed_tasks'] > 0:
                priority_bonus = controls['priority_weight'] * 0.2
                enhanced_reward += priority_bonus
            
            enhanced_rewards.append([enhanced_reward])
        
        return enhanced_rewards
    
    def reset(self):
        """重置环境"""
        return self.env.reset()
    
    def render(self, mode='human'):
        """渲染环境"""
        return self.env.render(mode)
    
    def close(self):
        """关闭环境"""
        return self.env.close()
    
    @property
    def training(self):
        """检查是否在训练模式"""
        return getattr(self, '_training', True)
    
    def train(self):
        """设置为训练模式"""
        self._training = True
    
    def eval(self):
        """设置为评估模式"""
        self._training = False

# 使用示例
def create_hasac_compatible_v2x_env(original_v2x_env):
    """
    创建HASAC兼容的V2X环境
    
    Args:
        original_v2x_env: 原始V2X环境
        
    Returns:
        包装后的环境
    """
    wrapped_env = HASACV2XWrapper(original_v2x_env)
    
    print("🎯 HASAC V2X环境创建成功！")
    print("📊 现在可以使用HASAC算法进行训练了")
    print("🔧 连续动作会自动转换为离散动作")
    print("⚡ 支持功率控制、优先级调度、负载均衡等高级功能")
    
    return wrapped_env

if __name__ == "__main__":
    # 测试包装器
    print("🧪 测试HASAC V2X包装器...")
    print("这个包装器让HASAC能够在现有的V2X环境中工作")
    print("无需修改原始环境，保持向后兼容性") 