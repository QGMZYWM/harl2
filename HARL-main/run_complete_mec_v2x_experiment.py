#!/usr/bin/env python3
"""
完整MEC-V2X仿真实验运行器

执行全面的MEC-V2X仿真实验，包括：
1. 多种场景测试
2. 性能对比分析
3. 详细的结果可视化
4. 实验报告生成
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime
from complete_mec_v2x_simulation import CompleteMECV2XSimulation, DEFAULT_CONFIG
from typing import Dict, List
import pandas as pd

class MECV2XExperimentRunner:
    """MEC-V2X实验运行器"""
    
    def __init__(self, base_config: Dict = None):
        """
        初始化实验运行器
        
        Args:
            base_config: 基础配置参数
        """
        self.base_config = base_config or DEFAULT_CONFIG.copy()
        self.results = {}
        self.experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"mec_v2x_experiment_results_{self.experiment_time}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ MEC-V2X实验运行器初始化完成")
        print(f"✓ 结果将保存到: {self.output_dir}")
    
    def run_comprehensive_experiment(self):
        """运行综合实验"""
        print("\n" + "="*60)
        print("开始综合MEC-V2X仿真实验")
        print("="*60)
        
        # 1. 基准实验
        print("\n1. 执行基准实验...")
        baseline_results = self.run_baseline_experiment()
        
        # 2. 车辆数量影响实验
        print("\n2. 执行车辆数量影响实验...")
        vehicle_scaling_results = self.run_vehicle_scaling_experiment()
        
        # 3. RSU密度影响实验
        print("\n3. 执行RSU密度影响实验...")
        rsu_density_results = self.run_rsu_density_experiment()
        
        # 4. 任务负载影响实验
        print("\n4. 执行任务负载影响实验...")
        task_load_results = self.run_task_load_experiment()
        
        # 5. 卸载策略对比实验
        print("\n5. 执行卸载策略对比实验...")
        strategy_comparison_results = self.run_strategy_comparison_experiment()
        
        # 保存所有结果
        self.results = {
            'baseline': baseline_results,
            'vehicle_scaling': vehicle_scaling_results,
            'rsu_density': rsu_density_results,
            'task_load': task_load_results,
            'strategy_comparison': strategy_comparison_results
        }
        
        # 生成报告和可视化
        self.generate_comprehensive_report()
        self.create_comprehensive_visualizations()
        
        print(f"\n✓ 实验完成！结果保存在: {self.output_dir}")
    
    def run_baseline_experiment(self) -> Dict:
        """运行基准实验"""
        config = self.base_config.copy()
        sim = CompleteMECV2XSimulation(config)
        
        return self._run_single_experiment(sim, "基准实验", steps=500)
    
    def run_vehicle_scaling_experiment(self) -> Dict:
        """车辆数量扩展实验"""
        vehicle_counts = [5, 10, 15, 20, 25]
        results = {}
        
        for count in vehicle_counts:
            print(f"  测试车辆数量: {count}")
            config = self.base_config.copy()
            config['num_vehicles'] = count
            
            sim = CompleteMECV2XSimulation(config)
            result = self._run_single_experiment(sim, f"车辆数={count}", steps=300)
            results[count] = result
        
        return results
    
    def run_rsu_density_experiment(self) -> Dict:
        """RSU密度实验"""
        rsu_counts = [2, 4, 6, 8]
        results = {}
        
        for count in rsu_counts:
            print(f"  测试RSU数量: {count}")
            config = self.base_config.copy()
            config['num_rsus'] = count
            
            sim = CompleteMECV2XSimulation(config)
            result = self._run_single_experiment(sim, f"RSU数={count}", steps=300)
            results[count] = result
        
        return results
    
    def run_task_load_experiment(self) -> Dict:
        """任务负载实验"""
        task_probs = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = {}
        
        for prob in task_probs:
            print(f"  测试任务生成概率: {prob}")
            config = self.base_config.copy()
            config['task_generation_prob'] = prob
            
            sim = CompleteMECV2XSimulation(config)
            result = self._run_single_experiment(sim, f"任务概率={prob}", steps=300)
            results[prob] = result
        
        return results
    
    def run_strategy_comparison_experiment(self) -> Dict:
        """卸载策略对比实验"""
        strategies = {
            'local_only': [1.0, 0.0, 0.0, 0.0, 0.5],      # 仅本地
            'rsu_preferred': [0.2, 0.6, 0.1, 0.1, 0.5],   # 偏好RSU
            'v2v_preferred': [0.2, 0.1, 0.6, 0.1, 0.5],   # 偏好V2V
            'cloud_preferred': [0.1, 0.1, 0.1, 0.7, 0.5], # 偏好云端
            'balanced': [0.25, 0.25, 0.25, 0.25, 0.5]     # 均衡策略
        }
        
        results = {}
        
        for strategy_name, action in strategies.items():
            print(f"  测试策略: {strategy_name}")
            config = self.base_config.copy()
            sim = CompleteMECV2XSimulation(config)
            
            # 运行固定策略实验
            result = self._run_fixed_strategy_experiment(sim, action, strategy_name, steps=300)
            results[strategy_name] = result
        
        return results
    
    def _run_single_experiment(self, sim: CompleteMECV2XSimulation, 
                              experiment_name: str, steps: int = 300) -> Dict:
        """运行单个实验"""
        observations = sim.reset()
        
        episode_rewards = []
        step_metrics = []
        
        total_reward = 0
        start_time = time.time()
        
        for step in range(steps):
            # 随机动作策略
            actions = {}
            for i in range(sim.config['num_vehicles']):
                actions[f'vehicle_{i}'] = np.random.uniform(0, 1, 5)
            
            obs, rewards, dones, info = sim.step(actions)
            
            step_reward = np.mean(list(rewards.values()))
            total_reward += step_reward
            episode_rewards.append(step_reward)
            
            # 收集步骤指标
            step_metric = {
                'step': step,
                'reward': step_reward,
                'active_connections': info['communication_stats']['active_connections'],
                'mec_utilization': info['communication_stats']['average_mec_utilization'],
                'completed_tasks': info['metrics']['completed_tasks'],
                'failed_tasks': info['metrics']['failed_tasks']
            }
            step_metrics.append(step_metric)
            
            if step % 50 == 0:
                print(f"    Step {step}: 奖励={step_reward:.4f}, 连接数={info['communication_stats']['active_connections']}")
        
        end_time = time.time()
        
        # 获取最终性能指标
        final_metrics = sim.get_performance_metrics()
        
        result = {
            'experiment_name': experiment_name,
            'total_steps': steps,
            'execution_time': end_time - start_time,
            'total_reward': total_reward,
            'average_reward': total_reward / steps,
            'episode_rewards': episode_rewards,
            'step_metrics': step_metrics,
            'final_metrics': final_metrics,
            'config': sim.config
        }
        
        print(f"    完成 - 平均奖励: {result['average_reward']:.4f}, 任务完成率: {final_metrics['task_completion_rate']:.4f}")
        
        return result
    
    def _run_fixed_strategy_experiment(self, sim: CompleteMECV2XSimulation,
                                     fixed_action: List[float], strategy_name: str,
                                     steps: int = 300) -> Dict:
        """运行固定策略实验"""
        observations = sim.reset()
        
        episode_rewards = []
        step_metrics = []
        
        total_reward = 0
        
        for step in range(steps):
            # 使用固定策略
            actions = {}
            for i in range(sim.config['num_vehicles']):
                actions[f'vehicle_{i}'] = np.array(fixed_action)
            
            obs, rewards, dones, info = sim.step(actions)
            
            step_reward = np.mean(list(rewards.values()))
            total_reward += step_reward
            episode_rewards.append(step_reward)
            
            step_metric = {
                'step': step,
                'reward': step_reward,
                'active_connections': info['communication_stats']['active_connections'],
                'mec_utilization': info['communication_stats']['average_mec_utilization'],
                'completed_tasks': info['metrics']['completed_tasks'],
                'failed_tasks': info['metrics']['failed_tasks']
            }
            step_metrics.append(step_metric)
        
        final_metrics = sim.get_performance_metrics()
        
        result = {
            'strategy_name': strategy_name,
            'fixed_action': fixed_action,
            'total_steps': steps,
            'total_reward': total_reward,
            'average_reward': total_reward / steps,
            'episode_rewards': episode_rewards,
            'step_metrics': step_metrics,
            'final_metrics': final_metrics
        }
        
        return result
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        report = {
            'experiment_info': {
                'timestamp': self.experiment_time,
                'base_config': self.base_config,
                'total_experiments': len(self.results)
            },
            'summary': {},
            'detailed_results': self.results
        }
        
        # 生成摘要
        if 'baseline' in self.results:
            baseline = self.results['baseline']
            report['summary']['baseline'] = {
                'average_reward': baseline['average_reward'],
                'task_completion_rate': baseline['final_metrics']['task_completion_rate'],
                'average_latency': baseline['final_metrics']['average_latency'],
                'average_energy': baseline['final_metrics']['average_energy_consumption']
            }
        
        # 车辆扩展性分析
        if 'vehicle_scaling' in self.results:
            scaling_summary = {}
            for count, result in self.results['vehicle_scaling'].items():
                scaling_summary[count] = {
                    'completion_rate': result['final_metrics']['task_completion_rate'],
                    'average_reward': result['average_reward']
                }
            report['summary']['vehicle_scaling'] = scaling_summary
        
        # 策略对比分析
        if 'strategy_comparison' in self.results:
            strategy_summary = {}
            for strategy, result in self.results['strategy_comparison'].items():
                strategy_summary[strategy] = {
                    'completion_rate': result['final_metrics']['task_completion_rate'],
                    'average_reward': result['average_reward'],
                    'average_latency': result['final_metrics']['average_latency']
                }
            report['summary']['strategy_comparison'] = strategy_summary
        
        # 保存报告
        report_file = os.path.join(self.output_dir, 'comprehensive_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✓ 综合报告已保存: {report_file}")
    
    def create_comprehensive_visualizations(self):
        """创建综合可视化"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 基准实验奖励曲线
        if 'baseline' in self.results:
            plt.subplot(3, 3, 1)
            baseline = self.results['baseline']
            plt.plot(baseline['episode_rewards'])
            plt.title('基准实验奖励曲线')
            plt.xlabel('步数')
            plt.ylabel('奖励')
            plt.grid(True)
        
        # 2. 车辆数量扩展性
        if 'vehicle_scaling' in self.results:
            plt.subplot(3, 3, 2)
            counts = []
            completion_rates = []
            avg_rewards = []
            
            for count, result in self.results['vehicle_scaling'].items():
                counts.append(count)
                completion_rates.append(result['final_metrics']['task_completion_rate'])
                avg_rewards.append(result['average_reward'])
            
            plt.plot(counts, completion_rates, 'o-', label='任务完成率')
            plt.plot(counts, [r/max(avg_rewards) for r in avg_rewards], 's-', label='归一化奖励')
            plt.title('车辆数量扩展性')
            plt.xlabel('车辆数量')
            plt.ylabel('性能指标')
            plt.legend()
            plt.grid(True)
        
        # 3. RSU密度影响
        if 'rsu_density' in self.results:
            plt.subplot(3, 3, 3)
            rsu_counts = []
            completion_rates = []
            
            for count, result in self.results['rsu_density'].items():
                rsu_counts.append(count)
                completion_rates.append(result['final_metrics']['task_completion_rate'])
            
            plt.plot(rsu_counts, completion_rates, 'o-')
            plt.title('RSU密度影响')
            plt.xlabel('RSU数量')
            plt.ylabel('任务完成率')
            plt.grid(True)
        
        # 4. 任务负载影响
        if 'task_load' in self.results:
            plt.subplot(3, 3, 4)
            task_probs = []
            latencies = []
            
            for prob, result in self.results['task_load'].items():
                task_probs.append(prob)
                latencies.append(result['final_metrics']['average_latency'])
            
            plt.plot(task_probs, latencies, 'o-')
            plt.title('任务负载影响')
            plt.xlabel('任务生成概率')
            plt.ylabel('平均延迟 (s)')
            plt.grid(True)
        
        # 5. 策略对比 - 完成率
        if 'strategy_comparison' in self.results:
            plt.subplot(3, 3, 5)
            strategies = []
            completion_rates = []
            
            for strategy, result in self.results['strategy_comparison'].items():
                strategies.append(strategy)
                completion_rates.append(result['final_metrics']['task_completion_rate'])
            
            bars = plt.bar(strategies, completion_rates)
            plt.title('卸载策略对比 - 任务完成率')
            plt.xlabel('策略')
            plt.ylabel('完成率')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, rate in zip(bars, completion_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
        
        # 6. 策略对比 - 延迟
        if 'strategy_comparison' in self.results:
            plt.subplot(3, 3, 6)
            strategies = []
            latencies = []
            
            for strategy, result in self.results['strategy_comparison'].items():
                strategies.append(strategy)
                latencies.append(result['final_metrics']['average_latency'])
            
            bars = plt.bar(strategies, latencies, color='orange')
            plt.title('卸载策略对比 - 平均延迟')
            plt.xlabel('策略')
            plt.ylabel('延迟 (s)')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, latency in zip(bars, latencies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{latency:.2f}', ha='center', va='bottom')
        
        # 7. 能耗对比
        if 'strategy_comparison' in self.results:
            plt.subplot(3, 3, 7)
            strategies = []
            energies = []
            
            for strategy, result in self.results['strategy_comparison'].items():
                strategies.append(strategy)
                energies.append(result['final_metrics']['average_energy_consumption'])
            
            plt.bar(strategies, energies, color='green')
            plt.title('卸载策略对比 - 平均能耗')
            plt.xlabel('策略')
            plt.ylabel('能耗 (J)')
            plt.xticks(rotation=45)
        
        # 8. V2V协作对比
        if 'strategy_comparison' in self.results:
            plt.subplot(3, 3, 8)
            strategies = []
            v2v_counts = []
            
            for strategy, result in self.results['strategy_comparison'].items():
                strategies.append(strategy)
                v2v_counts.append(result['final_metrics']['total_v2v_collaborations'])
            
            plt.bar(strategies, v2v_counts, color='purple')
            plt.title('V2V协作次数对比')
            plt.xlabel('策略')
            plt.ylabel('协作次数')
            plt.xticks(rotation=45)
        
        # 9. MEC利用率对比
        if 'strategy_comparison' in self.results:
            plt.subplot(3, 3, 9)
            strategies = []
            mec_utils = []
            
            for strategy, result in self.results['strategy_comparison'].items():
                strategies.append(strategy)
                mec_utils.append(result['final_metrics']['average_mec_utilization'])
            
            plt.bar(strategies, mec_utils, color='red')
            plt.title('MEC平均利用率对比')
            plt.xlabel('策略')
            plt.ylabel('利用率')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = os.path.join(self.output_dir, 'comprehensive_results.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 综合可视化已保存: {chart_file}")
        
        # 创建详细的策略对比表格
        self._create_strategy_comparison_table()
    
    def _create_strategy_comparison_table(self):
        """创建策略对比表格"""
        if 'strategy_comparison' not in self.results:
            return
        
        # 准备表格数据
        table_data = []
        for strategy, result in self.results['strategy_comparison'].items():
            row = {
                'Strategy': strategy,
                'Completion Rate': f"{result['final_metrics']['task_completion_rate']:.4f}",
                'Avg Latency (s)': f"{result['final_metrics']['average_latency']:.4f}",
                'Avg Energy (J)': f"{result['final_metrics']['average_energy_consumption']:.4f}",
                'V2V Collaborations': result['final_metrics']['total_v2v_collaborations'],
                'MEC Utilization': f"{result['final_metrics']['average_mec_utilization']:.4f}",
                'Avg Reward': f"{result['average_reward']:.4f}"
            }
            table_data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(table_data)
        table_file = os.path.join(self.output_dir, 'strategy_comparison_table.csv')
        df.to_csv(table_file, index=False)
        
        print(f"✓ 策略对比表格已保存: {table_file}")
    
    def print_experiment_summary(self):
        """打印实验摘要"""
        print("\n" + "="*60)
        print("MEC-V2X仿真实验结果摘要")
        print("="*60)
        
        if 'baseline' in self.results:
            baseline = self.results['baseline']
            print(f"\n基准实验结果:")
            print(f"  平均奖励: {baseline['average_reward']:.4f}")
            print(f"  任务完成率: {baseline['final_metrics']['task_completion_rate']:.4f}")
            print(f"  平均延迟: {baseline['final_metrics']['average_latency']:.4f}s")
            print(f"  平均能耗: {baseline['final_metrics']['average_energy_consumption']:.4f}J")
        
        if 'strategy_comparison' in self.results:
            print(f"\n最佳卸载策略分析:")
            best_completion = max(self.results['strategy_comparison'].items(),
                                key=lambda x: x[1]['final_metrics']['task_completion_rate'])
            best_latency = min(self.results['strategy_comparison'].items(),
                             key=lambda x: x[1]['final_metrics']['average_latency'])
            best_energy = min(self.results['strategy_comparison'].items(),
                            key=lambda x: x[1]['final_metrics']['average_energy_consumption'])
            
            print(f"  最高完成率: {best_completion[0]} ({best_completion[1]['final_metrics']['task_completion_rate']:.4f})")
            print(f"  最低延迟: {best_latency[0]} ({best_latency[1]['final_metrics']['average_latency']:.4f}s)")
            print(f"  最低能耗: {best_energy[0]} ({best_energy[1]['final_metrics']['average_energy_consumption']:.4f}J)")
        
        print(f"\n✓ 详细结果请查看: {self.output_dir}")

def main():
    """主函数"""
    print("完整MEC-V2X仿真实验系统")
    print("="*60)
    
    # 创建实验运行器
    runner = MECV2XExperimentRunner()
    
    # 运行综合实验
    start_time = time.time()
    runner.run_comprehensive_experiment()
    end_time = time.time()
    
    # 打印摘要
    runner.print_experiment_summary()
    
    print(f"\n总实验时间: {end_time - start_time:.2f} 秒")
    print("实验完成！")

if __name__ == "__main__":
    main() 