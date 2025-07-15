"""
V2X第一个创新点验证实验
验证动态上下文感知状态表征的效果

实验设置：
1. 基线对比：Standard HASAC vs HASAC+Transformer vs HASAC+Transformer+CL
2. 场景设置：不同动态性、车辆密度、任务负载
3. 评估指标：任务完成率、适应性、状态表征质量等
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import argparse
from typing import Dict, List, Tuple
import logging
import yaml # Added for loading YAML files

# 添加HARL路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.envs_tools import make_train_env, make_eval_env
from harl.runners.off_policy_ha_runner import OffPolicyHARunner


class V2XInnovation1Evaluator:
    """V2X第一个创新点验证器"""
    
    def __init__(self, base_config_path: str, result_dir: str = "experiment_results"):
        self.base_config_path = base_config_path
        self.result_dir = result_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录
        os.makedirs(result_dir, exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{result_dir}/experiment_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_experiment_configs(self) -> Dict[str, Dict]:
        """创建不同实验配置"""
        
        # 基础配置
        # 修改为直接读取YAML文件，而不是使用get_defaults_yaml_args
        try:
            with open(self.base_config_path, "r", encoding="utf-8") as file:
                base_args = yaml.load(file, Loader=yaml.FullLoader)
                self.logger.info(f"成功加载配置文件: {self.base_config_path}")
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            # 提供一个默认的基础配置
            base_args = {
                "env_name": "v2x",
                "algo": "hasac",
                "seed": 1,
                "cuda": True,
                "cuda_deterministic": False,
                "n_training_threads": 1,
                "n_rollout_threads": 8,
                "n_eval_rollout_threads": 1,
                "num_env_steps": 10000000,
                "episode_length": 200,
                "share_policy": True,
                "use_centralized_V": True,
                "hidden_size": 256,
                "layer_N": 2,
                "use_ReLU": True,
                "use_popart": False,
                "use_valuenorm": True,
                "use_feature_normalization": True,
                "use_orthogonal": True,
                "gain": 0.01,
                "use_naive_recurrent_policy": False,
                "use_recurrent_policy": False,
                "recurrent_N": 1,
                "data_chunk_length": 10,
                "lr": 5e-4,
                "critic_lr": 5e-4,
                "opti_eps": 1e-5,
                "weight_decay": 0,
                "save_interval": 1,
                "log_interval": 5,
                "use_eval": True,
                "eval_interval": 25,
                "eval_episodes": 32,
                "gamma": 0.99,
                "use_gae": True,
                "gae_lambda": 0.95,
                "use_proper_time_limits": False,
                "use_huber_loss": True,
                "huber_delta": 10.0,
                "use_value_active_masks": True,
                "use_policy_active_masks": True,
                "entropy_coef": 0.01,
                "value_loss_coef": 1,
                "use_max_grad_norm": True,
                "max_grad_norm": 10.0,
                "use_clipped_value_loss": True,
                "buffer_size": 5000,
                "batch_size": 256,
                "target_update_interval": 1,
                "target_update_tau": 0.005
            }
            self.logger.warning("使用默认配置")
        
        # 实验变体配置
        configs = {
            "baseline_hasac": {
                "name": "Standard HASAC",
                "description": "标准HASAC算法，不使用任何增强",
                "config": {
                    **base_args,
                    "use_transformer": False,
                    "use_contrastive_learning": False,
                    "exp_name": "baseline_hasac"
                }
            },
            
            "hasac_transformer": {
                "name": "HASAC + Transformer",
                "description": "HASAC + Transformer编码器，不使用对比学习",
                "config": {
                    **base_args,
                    "use_transformer": True,
                    "use_contrastive_learning": False,
                    "transformer_d_model": 256,
                    "transformer_nhead": 8,
                    "transformer_num_layers": 4,
                    "max_seq_length": 50,
                    "exp_name": "hasac_transformer"
                }
            },
            
            "hasac_transformer_cl": {
                "name": "HASAC + Transformer + CL",
                "description": "完整的第一个创新点：Transformer + 对比学习",
                "config": {
                    **base_args,
                    "use_transformer": True,
                    "use_contrastive_learning": True,
                    "transformer_d_model": 256,
                    "transformer_nhead": 8,
                    "transformer_num_layers": 4,
                    "max_seq_length": 50,
                    "contrastive_temperature": 0.1,
                    "similarity_threshold": 0.8,
                    "temporal_weight": 0.1,
                    "contrastive_loss_weight": 0.1,  # 修改参数名以匹配配置文件
                    "lambda_cl": 0.1,  # 保留lambda_cl参数以兼容现有代码
                    "exp_name": "hasac_transformer_cl"
                }
            }
        }
        
        return configs
    
    def create_scenario_configs(self) -> Dict[str, Dict]:
        """创建不同V2X场景配置"""
        
        scenarios = {
            "low_dynamics": {
                "name": "低动态场景",
                "description": "车辆移动缓慢，网络相对稳定",
                "env_args": {
                    "vehicle_speed_range": [10.0, 30.0],  # 低速
                    "task_generation_prob": 0.2,  # 低任务生成率
                    "num_agents": 8,
                    "communication_range": 400.0,  # 大通信范围
                    "max_episode_steps": 150
                }
            },
            
            "medium_dynamics": {
                "name": "中等动态场景", 
                "description": "标准V2X场景",
                "env_args": {
                    "vehicle_speed_range": [20.0, 60.0],  # 中等速度
                    "task_generation_prob": 0.3,  # 中等任务生成率
                    "num_agents": 10,
                    "communication_range": 300.0,  # 标准通信范围
                    "max_episode_steps": 200
                }
            },
            
            "high_dynamics": {
                "name": "高动态场景",
                "description": "高速移动，网络快速变化",
                "env_args": {
                    "vehicle_speed_range": [40.0, 100.0],  # 高速
                    "task_generation_prob": 0.4,  # 高任务生成率
                    "num_agents": 12,
                    "communication_range": 200.0,  # 小通信范围
                    "max_episode_steps": 250
                }
            },
            
            "dense_traffic": {
                "name": "密集交通场景",
                "description": "车辆密度大，资源竞争激烈",
                "env_args": {
                    "vehicle_speed_range": [15.0, 45.0],  # 拥堵速度
                    "task_generation_prob": 0.5,  # 高任务生成率
                    "num_agents": 15,  # 更多车辆
                    "communication_range": 250.0,
                    "max_episode_steps": 300
                }
            }
        }
        
        return scenarios
    
    def run_single_experiment(self, config: Dict, scenario: Dict, seed: int = 0) -> Dict:
        """运行单个实验"""
        
        self.logger.info(f"开始实验: {config['name']} 在 {scenario['name']}")
        
        # 合并配置
        args = config["config"].copy()
        
        # 确保scenario中有env_args
        env_args = scenario.get("env_args", {})
        if not env_args:
            self.logger.warning(f"场景 {scenario['name']} 没有提供env_args，使用默认值")
            # 提供默认的env_args
            env_args = {
                "vehicle_speed_range": [20.0, 60.0],
                "task_generation_prob": 0.3,
                "num_agents": 10,
                "communication_range": 300.0,
                "max_episode_steps": 200
            }
        
        # 更新配置
        args.update(env_args)
        args["seed"] = seed
        
        # 设置训练步数（用于快速验证）
        args["num_env_steps"] = 100000  # 可以根据需要调整
        args["eval_interval"] = 10000
        args["save_interval"] = 50000
        
        try:
            # 创建环境 - 将args同时作为env_args参数传递
            eval_envs = make_eval_env(args["env_name"], seed, 1, args)
            
            # 创建训练器
            runner = OffPolicyHARunner(args, eval_envs)
            
            # 保存runner引用以便后续可视化使用
            self.last_runner = runner
            
            # 训练
            results = runner.run()
            
            # 确保results是字典类型
            if results is None:
                results = {}
            
            # 提取关键指标
            metrics = self.extract_metrics(results, runner)
            
            # 清理
            eval_envs.close()
            
            self.logger.info(f"完成实验: {config['name']} 在 {scenario['name']}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"实验失败: {config['name']} 在 {scenario['name']}: {str(e)}")
            return {"error": str(e)}
    
    def extract_metrics(self, results: Dict, runner) -> Dict:
        """提取关键评估指标"""
        
        metrics = {
            "task_completion_rate": [],
            "task_failure_rate": [], 
            "energy_efficiency": [],
            "adaptation_speed": [],
            "state_representation_quality": 0.5,  # 默认值
            "episode_rewards": [],
            "convergence_steps": 0
        }
        
        # 从结果中提取指标（如果有）
        if results and isinstance(results, dict):
            # 从结果中提取任务完成率等指标
            metrics["task_completion_rate"] = results.get("task_completion_rate", [])
            metrics["task_failure_rate"] = results.get("task_failure_rate", [])
            metrics["energy_efficiency"] = results.get("energy_efficiency", [])
        
        # 从训练日志中提取指标
        if hasattr(runner, 'logger') and hasattr(runner.logger, 'episode_rewards'):
            rewards = runner.logger.episode_rewards
            if rewards and len(rewards) > 0:
                metrics["episode_rewards"] = rewards[-100:]  # 最后100个episode
        
        # 如果有对比学习，评估状态表征质量
        if hasattr(runner, 'actor') and hasattr(runner, 'num_agents'):
            if len(runner.actor) > 0:
                actor = runner.actor[0]
                if hasattr(actor, 'use_transformer') and actor.use_transformer:
                    metrics["state_representation_quality"] = self.evaluate_state_representation(actor)
        
        # 计算收敛速度
        if len(metrics["episode_rewards"]) > 0:
            metrics["convergence_steps"] = self.calculate_convergence_steps(metrics["episode_rewards"])
        
        # 计算平均值
        for key in ["task_completion_rate", "task_failure_rate", "energy_efficiency"]:
            if key in metrics and len(metrics[key]) > 0:
                metrics[f"avg_{key}"] = np.mean(metrics[key])
                metrics[f"std_{key}"] = np.std(metrics[key])
        
        return metrics
    
    def evaluate_state_representation(self, actor) -> float:
        """评估状态表征质量"""
        
        # 这里可以实现更复杂的状态表征质量评估
        # 例如：表征的可分离性、聚类质量等
        
        try:
            # 检查actor是否有必要的方法和属性
            if not hasattr(actor, 'compute_contrastive_loss') or not callable(getattr(actor, 'compute_contrastive_loss', None)):
                return 0.5  # 默认值，actor没有compute_contrastive_loss方法
                
            if not hasattr(actor, 'previous_contrastive_info') or actor.previous_contrastive_info is None:
                return 0.5  # 默认值，actor没有previous_contrastive_info属性或为None
            
            # 安全调用compute_contrastive_loss方法
            contrastive_loss = actor.compute_contrastive_loss()
            
            # 检查返回值是否为有效的tensor
            if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.numel() > 0 and contrastive_loss.item() > 0:
                return 1.0 / (1.0 + contrastive_loss.item())
        except Exception as e:
            # 记录错误但不中断评估流程
            self.logger.warning(f"评估状态表征时出错: {str(e)}")
        
        return 0.5  # 默认值
    
    def calculate_convergence_steps(self, rewards: List[float]) -> int:
        """计算收敛步数"""
        
        if len(rewards) < 10:
            return len(rewards) * 1000  # 估计值
        
        # 寻找奖励稳定的点
        window_size = 10
        threshold = 0.1
        
        for i in range(window_size, len(rewards)):
            recent_rewards = rewards[i-window_size:i]
            if np.std(recent_rewards) < threshold:
                return i * 1000  # 转换为训练步数
        
        return len(rewards) * 1000
    
    def run_comparative_experiments(self, num_seeds: int = 3) -> Dict:
        """运行对比实验"""
        
        self.logger.info("开始运行对比实验...")
        
        configs = self.create_experiment_configs()
        scenarios = self.create_scenario_configs()
        
        results = {
            "configs": configs,
            "scenarios": scenarios,
            "experiment_results": {},
            "summary": {}
        }
        
        # 运行所有配置和场景的组合
        for config_name, config in configs.items():
            results["experiment_results"][config_name] = {}
            
            for scenario_name, scenario in scenarios.items():
                self.logger.info(f"运行 {config_name} 在 {scenario_name}")
                
                seed_results = []
                for seed in range(num_seeds):
                    seed_result = self.run_single_experiment(config, scenario, seed)
                    seed_results.append(seed_result)
                
                # 聚合多个seed的结果
                aggregated = self.aggregate_seed_results(seed_results)
                results["experiment_results"][config_name][scenario_name] = aggregated
        
        # 生成摘要
        results["summary"] = self.generate_summary(results["experiment_results"])
        
        # 保存结果
        self.save_results(results)
        
        return results
    
    def aggregate_seed_results(self, seed_results: List[Dict]) -> Dict:
        """聚合多个seed的结果"""
        
        if not seed_results or all("error" in result for result in seed_results):
            return {"error": "所有seed都失败"}
        
        # 过滤错误结果
        valid_results = [r for r in seed_results if "error" not in r]
        if not valid_results:
            return {"error": "没有有效结果"}
        
        aggregated = {}
        
        # 聚合数值指标
        numeric_keys = ["avg_task_completion_rate", "avg_task_failure_rate", "avg_energy_efficiency", 
                       "convergence_steps", "state_representation_quality"]
        
        for key in numeric_keys:
            values = [r.get(key, 0) for r in valid_results if key in r]
            if values:
                aggregated[f"mean_{key}"] = np.mean(values)
                aggregated[f"std_{key}"] = np.std(values)
                aggregated[f"min_{key}"] = np.min(values)
                aggregated[f"max_{key}"] = np.max(values)
        
        # 聚合奖励曲线
        all_rewards = [r.get("episode_rewards", []) for r in valid_results]
        if all_rewards and all(len(rewards) > 0 for rewards in all_rewards):
            min_length = min(len(rewards) for rewards in all_rewards)
            truncated_rewards = [rewards[:min_length] for rewards in all_rewards]
            aggregated["mean_rewards"] = np.mean(truncated_rewards, axis=0).tolist()
            aggregated["std_rewards"] = np.std(truncated_rewards, axis=0).tolist()
        
        aggregated["num_successful_seeds"] = len(valid_results)
        aggregated["total_seeds"] = len(seed_results)
        
        return aggregated
    
    def generate_summary(self, experiment_results: Dict) -> Dict:
        """生成实验摘要"""
        
        summary = {
            "best_performer": {},
            "improvement_analysis": {},
            "scenario_analysis": {}
        }
        
        # 分析每个场景下的最佳表现者
        scenarios = list(next(iter(experiment_results.values())).keys())
        
        for scenario in scenarios:
            best_completion_rate = 0
            best_config = None
            
            scenario_results = {}
            for config_name, config_results in experiment_results.items():
                if scenario in config_results:
                    result = config_results[scenario]
                    completion_rate = result.get("mean_avg_task_completion_rate", 0)
                    scenario_results[config_name] = completion_rate
                    
                    if completion_rate > best_completion_rate:
                        best_completion_rate = completion_rate
                        best_config = config_name
            
            summary["best_performer"][scenario] = {
                "config": best_config,
                "completion_rate": best_completion_rate,
                "all_results": scenario_results
            }
        
        # 分析创新点的改进效果
        if "baseline_hasac" in experiment_results and "hasac_transformer_cl" in experiment_results:
            summary["improvement_analysis"] = self.analyze_improvement(
                experiment_results["baseline_hasac"],
                experiment_results["hasac_transformer_cl"]
            )
        
        return summary
    
    def analyze_improvement(self, baseline_results: Dict, enhanced_results: Dict) -> Dict:
        """分析创新点的改进效果"""
        
        improvements = {}
        
        for scenario in baseline_results.keys():
            if scenario in enhanced_results:
                baseline = baseline_results[scenario]
                enhanced = enhanced_results[scenario]
                
                # 计算改进百分比
                improvements[scenario] = {}
                
                metrics = ["mean_avg_task_completion_rate", "mean_convergence_steps", 
                          "mean_state_representation_quality"]
                
                for metric in metrics:
                    baseline_val = baseline.get(metric, 0)
                    enhanced_val = enhanced.get(metric, 0)
                    
                    if baseline_val > 0:
                        if "convergence_steps" in metric:
                            # 收敛步数越小越好
                            improvement = (baseline_val - enhanced_val) / baseline_val * 100
                        else:
                            # 其他指标越大越好
                            improvement = (enhanced_val - baseline_val) / baseline_val * 100
                        
                        improvements[scenario][metric] = improvement
        
        return improvements
    
    def save_results(self, results: Dict):
        """保存实验结果"""
        
        # 保存完整结果
        result_file = f"{self.result_dir}/experiment_results_{self.timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {result_file}")
        
        # 生成可视化
        self.generate_visualizations(results)
    
    def generate_visualizations(self, results: Dict):
        """生成可视化图表"""
        
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            experiment_results = results["experiment_results"]
            
            # 1. 任务完成率对比图
            self.plot_completion_rates(experiment_results)
            
            # 2. 收敛速度对比图
            self.plot_convergence_speed(experiment_results)
            
            # 3. 状态表征质量对比图
            self.plot_representation_quality(experiment_results)
            
            # 4. 综合性能雷达图
            self.plot_performance_radar(experiment_results)
            
            # 5. 注意力权重可视化（如果有）
            self.visualize_attention_weights()
            
            self.logger.info("可视化图表已生成")
            
        except Exception as e:
            self.logger.error(f"生成可视化时出错: {str(e)}")
            
    def visualize_attention_weights(self):
        """可视化Transformer注意力权重"""
        try:
            # 尝试从最后一个runner中获取注意力权重
            if not hasattr(self, 'last_runner') or self.last_runner is None:
                self.logger.info("没有可用的runner来获取注意力权重")
                return
                
            # 从runner中获取actor
            if not hasattr(self.last_runner, 'actor') or len(self.last_runner.actor) == 0:
                self.logger.info("runner中没有可用的actor")
                return
                
            actor = self.last_runner.actor[0]
            
            # 检查actor是否是TransformerEnhancedPolicy类型
            if not hasattr(actor, 'use_transformer') or not actor.use_transformer:
                self.logger.info("actor不是使用Transformer的策略")
                return
                
            # 检查actor是否有transformer_encoder
            if not hasattr(actor, 'transformer_encoder'):
                self.logger.info("actor中没有transformer_encoder")
                return
                
            # 检查transformer_encoder是否有last_attention_weights
            if not hasattr(actor.transformer_encoder, 'last_attention_weights'):
                self.logger.info("transformer_encoder中没有last_attention_weights")
                return
                
            attention_weights = actor.transformer_encoder.last_attention_weights
            
            # 检查注意力权重是否为None
            if attention_weights is None:
                self.logger.info("注意力权重为None，可能需要先运行一次前向传播")
                
                # 尝试运行一次前向传播来获取注意力权重
                try:
                    # 创建一个虚拟的观测和动作
                    dummy_obs = torch.zeros(1, 1, actor.obs_dim, device=actor.device)
                    dummy_action = torch.zeros(1, 1, actor.action_dim, device=actor.device)
                    
                    # 运行前向传播
                    with torch.no_grad():
                        actor.transformer_encoder(dummy_obs, dummy_action)
                    
                    # 再次获取注意力权重
                    attention_weights = actor.transformer_encoder.last_attention_weights
                    
                    if attention_weights is None:
                        self.logger.warning("即使在前向传播后，注意力权重仍为None")
                        return
                except Exception as e:
                    self.logger.error(f"尝试运行前向传播时出错: {str(e)}")
                    return
            
            # 检查注意力权重的类型和维度
            if not isinstance(attention_weights, torch.Tensor):
                self.logger.warning(f"注意力权重不是有效的张量，而是 {type(attention_weights)}")
                return
                
            # 将注意力权重移到CPU并转换为numpy数组
            attention_weights = attention_weights.detach().cpu().numpy()
            
            # 检查注意力权重的维度
            if len(attention_weights.shape) != 5:
                self.logger.warning(f"注意力权重的维度不是5，而是 {len(attention_weights.shape)}")
                return
                
            # 获取维度信息
            num_layers, batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # 检查批次大小
            if batch_size == 0:
                self.logger.warning("注意力权重的批次大小为0")
                return
                
            # 创建一个大图，显示每一层、每个头的注意力
            fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 3, num_layers * 3))
            
            # 如果只有一层或一个头，确保axes是二维的
            if num_layers == 1 and num_heads == 1:
                axes = np.array([[axes]])
            elif num_layers == 1:
                axes = np.array([axes])
            elif num_heads == 1:
                axes = axes.reshape(-1, 1)
            
            # 对于第一个批次样本，可视化所有层和头的注意力权重
            for layer in range(num_layers):
                for head in range(num_heads):
                    ax = axes[layer, head]
                    
                    # 获取当前层和头的注意力权重
                    attn = attention_weights[layer, 0, head]  # 使用第一个批次样本
                    
                    # 绘制热力图
                    im = ax.imshow(attn, cmap='viridis')
                    
                    # 设置标题和标签
                    ax.set_title(f'层 {layer+1}, 头 {head+1}')
                    ax.set_xlabel('序列位置 (目标)')
                    ax.set_ylabel('序列位置 (查询)')
                    
                    # 添加颜色条
                    fig.colorbar(im, ax=ax, shrink=0.8)
            
            plt.tight_layout()
            plt.savefig(f"{self.result_dir}/attention_weights_{self.timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 绘制平均注意力热力图
            plt.figure(figsize=(10, 8))
            
            # 计算所有层和头的平均注意力
            avg_attention = np.mean(attention_weights, axis=(0, 2))  # 平均所有层和头
            
            plt.imshow(avg_attention[0], cmap='viridis')
            plt.colorbar(shrink=0.8, label='注意力权重')
            plt.title('平均注意力权重热力图')
            plt.xlabel('序列位置 (目标)')
            plt.ylabel('序列位置 (查询)')
            
            plt.tight_layout()
            plt.savefig(f"{self.result_dir}/avg_attention_{self.timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("注意力权重可视化已保存")
            
        except Exception as e:
            self.logger.error(f"可视化注意力权重时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def plot_completion_rates(self, experiment_results: Dict):
        """绘制任务完成率对比图"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = list(experiment_results.keys())
        scenarios = list(next(iter(experiment_results.values())).keys())
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, config in enumerate(configs):
            completion_rates = []
            errors = []
            
            for scenario in scenarios:
                result = experiment_results[config].get(scenario, {})
                mean_rate = result.get("mean_avg_task_completion_rate", 0)
                std_rate = result.get("std_avg_task_completion_rate", 0)
                
                completion_rates.append(mean_rate)
                errors.append(std_rate)
            
            ax.bar(x + i * width, completion_rates, width, 
                  label=config, yerr=errors, capsize=5)
        
        ax.set_xlabel('实验场景')
        ax.set_ylabel('任务完成率')
        ax.set_title('不同算法在各场景下的任务完成率对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/completion_rates_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_speed(self, experiment_results: Dict):
        """绘制收敛速度对比图"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = list(experiment_results.keys())
        scenarios = list(next(iter(experiment_results.values())).keys())
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, config in enumerate(configs):
            convergence_steps = []
            errors = []
            
            for scenario in scenarios:
                result = experiment_results[config].get(scenario, {})
                mean_steps = result.get("mean_convergence_steps", 0)
                std_steps = result.get("std_convergence_steps", 0)
                
                convergence_steps.append(mean_steps / 1000)  # 转换为千步
                errors.append(std_steps / 1000)
            
            ax.bar(x + i * width, convergence_steps, width,
                  label=config, yerr=errors, capsize=5)
        
        ax.set_xlabel('实验场景')
        ax.set_ylabel('收敛步数 (千步)')
        ax.set_title('不同算法的收敛速度对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/convergence_speed_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_representation_quality(self, experiment_results: Dict):
        """绘制状态表征质量对比图"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = list(experiment_results.keys())
        scenarios = list(next(iter(experiment_results.values())).keys())
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, config in enumerate(configs):
            quality_scores = []
            errors = []
            
            for scenario in scenarios:
                result = experiment_results[config].get(scenario, {})
                mean_quality = result.get("mean_state_representation_quality", 0)
                std_quality = result.get("std_state_representation_quality", 0)
                
                quality_scores.append(mean_quality)
                errors.append(std_quality)
            
            ax.bar(x + i * width, quality_scores, width,
                  label=config, yerr=errors, capsize=5)
        
        ax.set_xlabel('实验场景')
        ax.set_ylabel('状态表征质量分数')
        ax.set_title('不同算法的状态表征质量对比')
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.result_dir}/representation_quality_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_radar(self, experiment_results: Dict):
        """绘制综合性能雷达图"""
        
        # 这里实现雷达图绘制逻辑
        # 由于空间限制，简化实现
        self.logger.info("雷达图功能待实现")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="V2X第一个创新点验证实验")
    parser.add_argument("--config", type=str, default="harl_innovation1_config.yaml",
                       help="基础配置文件路径")
    parser.add_argument("--result_dir", type=str, default="v2x_innovation1_results",
                       help="结果保存目录")
    parser.add_argument("--num_seeds", type=int, default=3,
                       help="每个实验运行的种子数")
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = V2XInnovation1Evaluator(args.config, args.result_dir)
    
    # 运行实验
    results = evaluator.run_comparative_experiments(args.num_seeds)
    
    # 打印摘要
    print("\n" + "="*50)
    print("实验摘要")
    print("="*50)
    
    summary = results["summary"]
    if "best_performer" in summary:
        for scenario, best in summary["best_performer"].items():
            print(f"\n{scenario}场景最佳算法: {best['config']}")
            print(f"任务完成率: {best['completion_rate']:.4f}")
    
    if "improvement_analysis" in summary:
        print(f"\n创新点改进分析:")
        for scenario, improvements in summary["improvement_analysis"].items():
            print(f"\n{scenario}场景:")
            for metric, improvement in improvements.items():
                print(f"  {metric}: {improvement:.2f}%")


if __name__ == "__main__":
    main() 