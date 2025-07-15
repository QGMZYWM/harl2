"""
HASAC-Flow MEC-V2X训练脚本

基于完整MEC-V2X融合环境的训练示例
集成Transformer状态表征、角色分配、多目标奖励等创新功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
import os
from datetime import datetime
import random

from hasac_flow_mec_v2x_env import HASACFlowMECEnv, VehicleRole


class HASACFlowTrainer:
    """HASAC-Flow训练器"""
    
    def __init__(self, config):
        self.config = config
        
        # 创建环境
        self.env = HASACFlowMECEnv(config)
        
        # 创建日志目录
        self.log_dir = f"logs/hasac_flow_mec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard写入器
        self.writer = SummaryWriter(self.log_dir)
        
        # 训练参数
        self.num_episodes = config.get('num_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 200)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 100000)
        
        # 创建优化器
        self.optimizers = self._create_optimizers()
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # 性能指标
        self.metrics = {
            'episode_rewards': [],
            'task_completion_rates': [],
            'energy_efficiency': [],
            'collaboration_quality': [],
            'security_scores': [],
            'role_distributions': defaultdict(list)
        }
        
        # 对比学习标签生成器
        self.label_generator = ContextLabelGenerator()
        
        # 全局步数计数器
        self.global_step = 0
    
    def _create_optimizers(self):
        """创建优化器"""
        optimizers = {}
        
        # Transformer编码器优化器
        optimizers['transformer'] = optim.Adam(
            self.env.transformer_encoder.parameters(),
            lr=self.learning_rate
        )
        
        # 对比学习优化器
        optimizers['contrastive'] = optim.Adam(
            self.env.contrastive_module.parameters(),
            lr=self.learning_rate
        )
        
        # 角色分配网络优化器
        optimizers['role_assignment'] = optim.Adam(
            self.env.role_assignment_network.parameters(),
            lr=self.learning_rate
        )
        
        # Kaleidoscope策略网络优化器
        optimizers['policy'] = optim.Adam(
            self.env.kaleidoscope_policy.parameters(),
            lr=self.learning_rate
        )
        
        return optimizers
    
    def train(self):
        """主训练循环"""
        print(f"开始训练HASAC-Flow MEC-V2X环境")
        print(f"配置: {self.config}")
        print(f"日志目录: {self.log_dir}")
        
        for episode in range(self.num_episodes):
            episode_metrics = self._train_episode(episode)
            
            # 记录指标
            self._record_metrics(episode, episode_metrics)
            
            # 定期保存模型
            if episode % 100 == 0:
                self._save_checkpoint(episode)
            
            # 定期评估
            if episode % 50 == 0:
                eval_metrics = self._evaluate()
                self._log_evaluation(episode, eval_metrics)
            
            # 打印进度
            if episode % 10 == 0:
                print(f"Episode {episode}/{self.num_episodes}")
                print(f"  平均奖励: {np.mean(self.metrics['episode_rewards'][-10:]):.3f}")
                print(f"  任务完成率: {np.mean(self.metrics['task_completion_rates'][-10:]):.3f}")
                print(f"  能效: {np.mean(self.metrics['energy_efficiency'][-10:]):.3f}")
        
        # 训练结束
        self._save_final_results()
        self.writer.close()
        
        print(f"训练完成！结果保存在: {self.log_dir}")
    
    def _train_episode(self, episode):
        """训练单个episode"""
        # 重置环境
        observations = self.env.reset()
        
        episode_rewards = []
        episode_actions = []
        episode_states = []
        role_counts = defaultdict(int)
        
        for step in range(self.max_steps_per_episode):
            # 生成动作
            actions = self._generate_actions(observations)
            
            # 执行动作
            next_observations, rewards, dones, infos = self.env.step(actions)
            
            # 存储经验
            experience = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'next_observations': next_observations,
                'dones': dones,
                'infos': infos
            }
            self.replay_buffer.append(experience)
            
            # 训练网络
            if len(self.replay_buffer) >= self.batch_size:
                self._train_networks()
            
            # 记录统计
            episode_rewards.append(sum(rewards.values()))
            episode_actions.append(actions)
            episode_states.append(observations)
            
            # 统计角色分布
            self._count_roles(role_counts)
            
            # 更新观测
            observations = next_observations
            
            # 检查终止条件
            if all(dones.values()):
                break
        
        # 计算episode指标
        episode_metrics = {
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards),
            'task_completion_rate': self._calculate_task_completion_rate(infos),
            'energy_efficiency': self._calculate_energy_efficiency(infos),
            'collaboration_quality': self._calculate_collaboration_quality(infos),
            'security_score': self._calculate_security_score(infos),
            'role_distribution': role_counts
        }
        
        return episode_metrics
    
    def _generate_actions(self, observations):
        """生成动作"""
        actions = {}
        
        # 将观测转换为状态嵌入
        state_embeddings = self._compute_state_embeddings(observations)
        
        # 角色分配
        role_assignments = self._assign_roles(state_embeddings)
        
        # 生成策略
        for agent_id in observations.keys():
            state_embedding = state_embeddings[agent_id]
            role_probs = role_assignments[agent_id]
            
            # 使用Kaleidoscope策略生成动作
            with torch.no_grad():
                action_logits = self.env.kaleidoscope_policy(
                    state_embedding.unsqueeze(0),
                    role_probs.unsqueeze(0)
                )
                
                # 添加探索噪声
                noise = torch.randn_like(action_logits) * 0.1
                action = torch.tanh(action_logits + noise)
                
                actions[agent_id] = action.squeeze(0).numpy()
        
        return actions
    
    def _compute_state_embeddings(self, observations):
        """计算状态嵌入"""
        state_embeddings = {}
        
        for agent_id, obs in observations.items():
            # 获取历史序列
            history = list(self.env.history_buffer[agent_id])
            
            if len(history) >= self.env.sequence_length:
                # 转换为tensor
                history_tensor = torch.FloatTensor(history[-self.env.sequence_length:]).unsqueeze(0)
                
                # Transformer编码
                state_embedding = self.env.transformer_encoder(history_tensor)
                state_embeddings[agent_id] = state_embedding.squeeze(0)
            else:
                # 使用零填充
                state_embeddings[agent_id] = torch.zeros(self.env.hidden_dim)
        
        return state_embeddings
    
    def _assign_roles(self, state_embeddings):
        """分配角色"""
        role_assignments = {}
        
        for agent_id, embedding in state_embeddings.items():
            role_probs = self.env.role_assignment_network(embedding.unsqueeze(0))
            role_assignments[agent_id] = role_probs.squeeze(0)
        
        return role_assignments
    
    def _train_networks(self):
        """训练网络"""
        # 从经验回放缓冲区采样
        batch = self._sample_batch()
        
        # 训练Transformer编码器和对比学习
        self._train_state_representation(batch)
        
        # 训练角色分配网络
        self._train_role_assignment(batch)
        
        # 训练Kaleidoscope策略网络
        self._train_policy_network(batch)
    
    def _sample_batch(self):
        """采样训练批次"""
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, batch_size)
        return batch
    
    def _train_state_representation(self, batch):
        """训练状态表征（Transformer + 对比学习）"""
        # 准备对比学习数据
        embeddings = []
        labels = []
        
        for experience in batch:
            for agent_id, obs in experience['observations'].items():
                # 获取历史序列
                history = list(self.env.history_buffer[agent_id])
                
                if len(history) >= self.env.sequence_length:
                    history_tensor = torch.FloatTensor(history[-self.env.sequence_length:]).unsqueeze(0)
                    embedding = self.env.transformer_encoder(history_tensor)
                    embeddings.append(embedding.squeeze(0))
                    
                    # 生成对比学习标签
                    label = self.label_generator.generate_label(obs, experience['infos'])
                    labels.append(label)
        
        if embeddings:
            embeddings = torch.stack(embeddings)
            labels = torch.tensor(labels)
            
            # 计算对比学习损失
            contrastive_loss = self.env.contrastive_module(embeddings, labels)
            
            # 反向传播
            self.optimizers['transformer'].zero_grad()
            self.optimizers['contrastive'].zero_grad()
            
            contrastive_loss.backward()
            
            self.optimizers['transformer'].step()
            self.optimizers['contrastive'].step()
            
            # 记录损失
            self.writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), self.global_step)
    
    def _train_role_assignment(self, batch):
        """训练角色分配网络"""
        # 这里可以实现角色分配的监督学习
        # 基于任务类型、车辆状态等信息预测最优角色
        pass
    
    def _train_policy_network(self, batch):
        """训练策略网络"""
        # 这里可以实现策略梯度更新
        # 基于奖励信号优化Kaleidoscope策略
        pass
    
    def _count_roles(self, role_counts):
        """统计角色分布"""
        for agent_id in range(self.env.num_agents):
            # 获取当前角色分配
            obs = self.env._get_observations()[agent_id]
            state_embedding = self._compute_state_embeddings({agent_id: obs})[agent_id]
            role_probs = self.env.role_assignment_network(state_embedding.unsqueeze(0))
            
            # 找到最可能的角色
            dominant_role = torch.argmax(role_probs).item()
            role_name = list(VehicleRole)[dominant_role].value
            role_counts[role_name] += 1
    
    def _calculate_task_completion_rate(self, infos):
        """计算任务完成率"""
        total_tasks = 0
        completed_tasks = 0
        
        for agent_id, vehicle in self.env.vehicles.items():
            total_tasks += len(vehicle.tasks)
            # 简化：假设正在处理的任务都会完成
            completed_tasks += max(0, len(vehicle.tasks) - 1)
        
        return completed_tasks / max(1, total_tasks)
    
    def _calculate_energy_efficiency(self, infos):
        """计算能效"""
        total_energy = sum(v.energy_consumption_rate for v in self.env.vehicles.values())
        total_tasks = sum(len(v.tasks) for v in self.env.vehicles.values())
        
        return total_tasks / max(1, total_energy)
    
    def _calculate_collaboration_quality(self, infos):
        """计算协作质量"""
        # 基于V2V和V2I交互的成功率
        return 0.8  # 简化实现
    
    def _calculate_security_score(self, infos):
        """计算安全分数"""
        trust_scores = []
        for agent_id in range(self.env.num_agents):
            trust_score = self.env.security_module.evaluate_trust(agent_id, agent_id)
            trust_scores.append(trust_score)
        
        return np.mean(trust_scores)
    
    def _record_metrics(self, episode, metrics):
        """记录指标"""
        self.metrics['episode_rewards'].append(metrics['total_reward'])
        self.metrics['task_completion_rates'].append(metrics['task_completion_rate'])
        self.metrics['energy_efficiency'].append(metrics['energy_efficiency'])
        self.metrics['collaboration_quality'].append(metrics['collaboration_quality'])
        self.metrics['security_scores'].append(metrics['security_score'])
        
        # 记录角色分布
        for role, count in metrics['role_distribution'].items():
            self.metrics['role_distributions'][role].append(count)
        
        # TensorBoard记录
        self.writer.add_scalar('Reward/Episode', metrics['total_reward'], episode)
        self.writer.add_scalar('Performance/TaskCompletion', metrics['task_completion_rate'], episode)
        self.writer.add_scalar('Performance/EnergyEfficiency', metrics['energy_efficiency'], episode)
        self.writer.add_scalar('Performance/CollaborationQuality', metrics['collaboration_quality'], episode)
        self.writer.add_scalar('Security/TrustScore', metrics['security_score'], episode)
        
        # 记录角色分布
        for role, count in metrics['role_distribution'].items():
            self.writer.add_scalar(f'Roles/{role}', count, episode)
    
    def _evaluate(self):
        """评估模型"""
        eval_rewards = []
        eval_completion_rates = []
        
        for _ in range(10):  # 评估10个episode
            obs = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                actions = self._generate_actions(obs)
                obs, rewards, dones, infos = self.env.step(actions)
                episode_reward += sum(rewards.values())
                
                if all(dones.values()):
                    break
            
            eval_rewards.append(episode_reward)
            eval_completion_rates.append(self._calculate_task_completion_rate(infos))
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_completion_rate': np.mean(eval_completion_rates)
        }
    
    def _log_evaluation(self, episode, eval_metrics):
        """记录评估结果"""
        print(f"  评估结果 (Episode {episode}):")
        print(f"    平均奖励: {eval_metrics['avg_reward']:.3f}")
        print(f"    平均完成率: {eval_metrics['avg_completion_rate']:.3f}")
        
        self.writer.add_scalar('Eval/AvgReward', eval_metrics['avg_reward'], episode)
        self.writer.add_scalar('Eval/AvgCompletionRate', eval_metrics['avg_completion_rate'], episode)
    
    def _save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'transformer_state': self.env.transformer_encoder.state_dict(),
            'contrastive_state': self.env.contrastive_module.state_dict(),
            'role_assignment_state': self.env.role_assignment_network.state_dict(),
            'kaleidoscope_state': self.env.kaleidoscope_policy.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, f"{self.log_dir}/checkpoint_episode_{episode}.pt")
    
    def _save_final_results(self):
        """保存最终结果"""
        # 保存指标
        with open(f"{self.log_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # 保存最终模型
        final_model = {
            'transformer': self.env.transformer_encoder.state_dict(),
            'contrastive': self.env.contrastive_module.state_dict(),
            'role_assignment': self.env.role_assignment_network.state_dict(),
            'kaleidoscope': self.env.kaleidoscope_policy.state_dict(),
            'config': self.config
        }
        
        torch.save(final_model, f"{self.log_dir}/final_model.pt")
        
        # 生成性能图表
        self._generate_plots()
    
    def _generate_plots(self):
        """生成性能图表"""
        plt.figure(figsize=(15, 10))
        
        # 奖励曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.metrics['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # 任务完成率
        plt.subplot(2, 3, 2)
        plt.plot(self.metrics['task_completion_rates'])
        plt.title('Task Completion Rate')
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        
        # 能效
        plt.subplot(2, 3, 3)
        plt.plot(self.metrics['energy_efficiency'])
        plt.title('Energy Efficiency')
        plt.xlabel('Episode')
        plt.ylabel('Efficiency')
        
        # 协作质量
        plt.subplot(2, 3, 4)
        plt.plot(self.metrics['collaboration_quality'])
        plt.title('Collaboration Quality')
        plt.xlabel('Episode')
        plt.ylabel('Quality')
        
        # 安全分数
        plt.subplot(2, 3, 5)
        plt.plot(self.metrics['security_scores'])
        plt.title('Security Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # 角色分布
        plt.subplot(2, 3, 6)
        for role, counts in self.metrics['role_distributions'].items():
            plt.plot(counts, label=role)
        plt.title('Role Distribution')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/performance_plots.png")
        plt.close()


class ContextLabelGenerator:
    """上下文标签生成器用于对比学习"""
    
    def __init__(self):
        self.mobility_patterns = ['highway', 'urban', 'parking']
        self.load_levels = ['low', 'medium', 'high']
        self.battery_levels = ['low', 'medium', 'high']
    
    def generate_label(self, observation, info):
        """生成对比学习标签"""
        # 基于观测生成上下文标签
        battery_level = observation[4]  # 电池水平
        load_level = observation[5]     # 负载水平
        
        # 简化的标签生成逻辑
        if battery_level < 0.3:
            battery_cat = 0  # 低电量
        elif battery_level < 0.7:
            battery_cat = 1  # 中等电量
        else:
            battery_cat = 2  # 高电量
        
        if load_level < 0.3:
            load_cat = 0  # 低负载
        elif load_level < 0.7:
            load_cat = 1  # 中等负载
        else:
            load_cat = 2  # 高负载
        
        # 组合标签
        label = battery_cat * 3 + load_cat
        
        return label


def main():
    """主函数"""
    # 配置参数
    config = {
        # 环境参数
        'num_agents': 10,
        'num_rsus': 4,
        'map_size': 1000,
        'sequence_length': 10,
        'max_episode_steps': 200,
        'state_dim': 64,
        'hidden_dim': 256,
        
        # 训练参数
        'num_episodes': 1000,
        'max_steps_per_episode': 200,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 64,
        'buffer_size': 100000
    }
    
    # 创建训练器
    trainer = HASACFlowTrainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main() 