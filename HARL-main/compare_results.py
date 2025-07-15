#!/usr/bin/env python3
"""
比较不同实验模式的结果脚本
用于离线比较基准模式、增强模式和消融研究模式的结果
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tabulate import tabulate
from scipy import stats

def find_result_directories(base_dir="logs"):
    """查找所有结果目录"""
    # 查找所有模式的结果目录
    baseline_dirs = glob.glob(f"{base_dir}/baseline_*_run_*")
    enhanced_dirs = glob.glob(f"{base_dir}/enhanced_full_run_*")
    transformer_only_dirs = glob.glob(f"{base_dir}/enhanced_transformer_only_run_*")
    contrastive_only_dirs = glob.glob(f"{base_dir}/enhanced_contrastive_only_run_*")
    
    # 按时间戳排序（最新的在前）
    baseline_dirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    enhanced_dirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    transformer_only_dirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    contrastive_only_dirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
    
    return {
        'baseline': baseline_dirs,
        'enhanced': enhanced_dirs,
        'transformer_only': transformer_only_dirs,
        'contrastive_only': contrastive_only_dirs
    }

def load_report(dir_path):
    """加载验证报告"""
    report_path = os.path.join(dir_path, 'validation_report.json')
    if not os.path.exists(report_path):
        print(f"警告：{dir_path}中未找到验证报告")
        return None
    
    with open(report_path, 'r') as f:
        return json.load(f)

def load_learning_curves(dir_path):
    """加载学习曲线数据"""
    rewards_path = os.path.join(dir_path, 'rewards.png')
    transformer_path = os.path.join(dir_path, 'transformer_effectiveness.png')
    contrastive_path = os.path.join(dir_path, 'contrastive_loss.png')
    
    # 检查文件是否存在
    has_rewards = os.path.exists(rewards_path)
    has_transformer = os.path.exists(transformer_path)
    has_contrastive = os.path.exists(contrastive_path)
    
    # 从TensorBoard日志中加载数据（如果可用）
    # 这里简化处理，实际上需要使用tensorboard的API来读取
    
    return {
        'has_rewards': has_rewards,
        'has_transformer': has_transformer,
        'has_contrastive': has_contrastive,
        'rewards_path': rewards_path if has_rewards else None,
        'transformer_path': transformer_path if has_transformer else None,
        'contrastive_path': contrastive_path if has_contrastive else None
    }

def compare_results(dirs_dict, output_dir="comparison_results"):
    """比较不同模式的结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有报告
    reports = {}
    for mode, dirs in dirs_dict.items():
        if dirs:
            report = load_report(dirs[0])
            if report:
                reports[mode] = report
    
    if not reports:
        print("错误：未找到有效的验证报告")
        return
    
    # 1. 生成表格比较
    generate_comparison_table(reports, output_dir)
    
    # 2. 生成性能对比图
    generate_performance_comparison(reports, output_dir)
    
    # 3. 生成消融研究图
    generate_ablation_study_chart(reports, output_dir)
    
    # 4. 进行统计显著性测试（如果有多次运行的数据）
    perform_statistical_tests(dirs_dict, output_dir)
    
    print(f"\n✓ 比较结果已保存到: {output_dir}")

def generate_comparison_table(reports, output_dir):
    """生成比较表格"""
    # 准备表格数据
    headers = ["指标", "基准模式", "仅Transformer", "仅对比学习", "完整增强模式"]
    rows = []
    
    # 平均奖励
    row = ["平均奖励"]
    for mode in ['baseline', 'transformer_only', 'contrastive_only', 'enhanced']:
        if mode in reports:
            row.append(f"{reports[mode].get('avg_reward', 'N/A'):.4f}")
        else:
            row.append("N/A")
    rows.append(row)
    
    # 平均episode长度
    row = ["平均Episode长度"]
    for mode in ['baseline', 'transformer_only', 'contrastive_only', 'enhanced']:
        if mode in reports:
            row.append(f"{reports[mode].get('avg_episode_length', 'N/A'):.2f}")
        else:
            row.append("N/A")
    rows.append(row)
    
    # Transformer有效性（如果适用）
    row = ["Transformer有效性"]
    for mode in ['baseline', 'transformer_only', 'contrastive_only', 'enhanced']:
        if mode in reports and 'transformer_effectiveness' in reports[mode]:
            row.append(f"{reports[mode]['transformer_effectiveness']:.4f}")
        else:
            row.append("N/A")
    rows.append(row)
    
    # 对比学习损失（如果适用）
    row = ["对比学习损失"]
    for mode in ['baseline', 'transformer_only', 'contrastive_only', 'enhanced']:
        if mode in reports and 'contrastive_loss' in reports[mode]:
            row.append(f"{reports[mode]['contrastive_loss']:.4f}")
        else:
            row.append("N/A")
    rows.append(row)
    
    # 生成表格
    table = tabulate(rows, headers, tablefmt="grid")
    
    # 保存到文件
    with open(os.path.join(output_dir, "comparison_table.txt"), "w") as f:
        f.write(table)
    
    # 打印表格
    print("\n性能比较表格：")
    print(table)

def generate_performance_comparison(reports, output_dir):
    """生成性能对比图"""
    modes = ['baseline', 'transformer_only', 'contrastive_only', 'enhanced']
    available_modes = [mode for mode in modes if mode in reports]
    
    if not available_modes:
        return
    
    # 平均奖励对比
    rewards = [reports[mode].get('avg_reward', 0) for mode in available_modes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(available_modes, rewards)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('不同模式的平均奖励对比')
    plt.ylabel('平均奖励')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'))
    plt.close()
    
    # 如果有Transformer有效性数据，也生成对比图
    if any('transformer_effectiveness' in reports.get(mode, {}) for mode in available_modes):
        transformer_effectiveness = []
        transformer_modes = []
        
        for mode in available_modes:
            if 'transformer_effectiveness' in reports[mode]:
                transformer_effectiveness.append(reports[mode]['transformer_effectiveness'])
                transformer_modes.append(mode)
        
        if transformer_effectiveness:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(transformer_modes, transformer_effectiveness)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.4f}', ha='center', va='bottom')
            
            plt.title('不同模式的Transformer有效性对比')
            plt.ylabel('Transformer有效性得分')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, 'transformer_effectiveness_comparison.png'))
            plt.close()

def generate_ablation_study_chart(reports, output_dir):
    """生成消融研究图表"""
    # 检查是否有足够的数据进行消融研究
    required_modes = ['baseline', 'transformer_only', 'enhanced']
    if not all(mode in reports for mode in required_modes):
        print("警告：缺少消融研究所需的模式数据")
        return
    
    # 提取奖励数据
    baseline_reward = reports['baseline'].get('avg_reward', 0)
    transformer_only_reward = reports['transformer_only'].get('avg_reward', 0)
    enhanced_reward = reports['enhanced'].get('avg_reward', 0)
    
    # 计算每个组件的贡献
    transformer_contribution = transformer_only_reward - baseline_reward
    contrastive_contribution = enhanced_reward - transformer_only_reward
    
    # 创建堆叠柱状图
    plt.figure(figsize=(10, 6))
    
    # 基准值
    plt.bar(['增强模式'], [baseline_reward], label='基准性能')
    
    # Transformer贡献
    plt.bar(['增强模式'], [transformer_contribution], bottom=[baseline_reward], 
            label='Transformer贡献')
    
    # 对比学习贡献
    plt.bar(['增强模式'], [contrastive_contribution], 
            bottom=[baseline_reward + transformer_contribution], 
            label='对比学习贡献')
    
    # 添加标签和图例
    plt.ylabel('平均奖励')
    plt.title('创新点1各组件贡献的消融研究')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    plt.text(0, baseline_reward/2, f'{baseline_reward:.4f}', 
             ha='center', va='center')
    plt.text(0, baseline_reward + transformer_contribution/2, 
             f'+{transformer_contribution:.4f}', ha='center', va='center')
    plt.text(0, baseline_reward + transformer_contribution + contrastive_contribution/2, 
             f'+{contrastive_contribution:.4f}', ha='center', va='center')
    
    plt.savefig(os.path.join(output_dir, 'ablation_study.png'))
    plt.close()
    
    # 创建饼图显示各组件的相对贡献
    total_improvement = enhanced_reward - baseline_reward
    if total_improvement > 0:
        plt.figure(figsize=(8, 8))
        
        # 计算百分比
        transformer_percent = (transformer_contribution / total_improvement) * 100
        contrastive_percent = (contrastive_contribution / total_improvement) * 100
        
        # 创建饼图
        plt.pie([transformer_percent, contrastive_percent], 
                labels=['Transformer', '对比学习'], 
                autopct='%1.1f%%',
                startangle=90,
                colors=['#ff9999','#66b3ff'])
        
        plt.axis('equal')  # 保持饼图为圆形
        plt.title('创新点1各组件的相对贡献')
        plt.savefig(os.path.join(output_dir, 'component_contribution_pie.png'))
        plt.close()

def perform_statistical_tests(dirs_dict, output_dir):
    """进行统计显著性测试"""
    # 如果每种模式有多次运行的结果，可以进行统计检验
    # 这里简化处理，只检查是否有足够的数据
    
    baseline_reports = []
    enhanced_reports = []
    
    for dir_path in dirs_dict.get('baseline', []):
        report = load_report(dir_path)
        if report:
            baseline_reports.append(report)
    
    for dir_path in dirs_dict.get('enhanced', []):
        report = load_report(dir_path)
        if report:
            enhanced_reports.append(report)
    
    # 如果两种模式都有至少3次运行的结果，进行t检验
    if len(baseline_reports) >= 3 and len(enhanced_reports) >= 3:
        baseline_rewards = [r.get('avg_reward', 0) for r in baseline_reports]
        enhanced_rewards = [r.get('avg_reward', 0) for r in enhanced_reports]
        
        t_stat, p_value = stats.ttest_ind(baseline_rewards, enhanced_rewards)
        
        # 保存结果
        with open(os.path.join(output_dir, "statistical_test.txt"), "w") as f:
            f.write("统计显著性测试结果：\n\n")
            f.write(f"基准模式运行次数: {len(baseline_rewards)}\n")
            f.write(f"增强模式运行次数: {len(enhanced_rewards)}\n")
            f.write(f"基准模式平均奖励: {np.mean(baseline_rewards):.4f} ± {np.std(baseline_rewards):.4f}\n")
            f.write(f"增强模式平均奖励: {np.mean(enhanced_rewards):.4f} ± {np.std(enhanced_rewards):.4f}\n")
            f.write(f"t统计量: {t_stat:.4f}\n")
            f.write(f"p值: {p_value:.4f}\n")
            f.write(f"结论: {'有统计显著差异' if p_value < 0.05 else '无统计显著差异'} (α=0.05)\n")
        
        print("\n统计显著性测试结果：")
        print(f"基准模式 vs 增强模式: p值 = {p_value:.4f}")
        print(f"结论: {'有统计显著差异' if p_value < 0.05 else '无统计显著差异'} (α=0.05)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较不同实验模式的结果")
    parser.add_argument("--logs_dir", default="logs", help="日志目录路径")
    parser.add_argument("--output_dir", default="comparison_results", help="输出目录路径")
    args = parser.parse_args()
    
    print("="*60)
    print("HARL创新点1验证结果比较工具")
    print("比较基准模式、增强模式和消融研究模式的结果")
    print("="*60)
    
    # 查找结果目录
    dirs_dict = find_result_directories(args.logs_dir)
    
    # 检查是否找到结果
    found_results = False
    for mode, dirs in dirs_dict.items():
        if dirs:
            found_results = True
            print(f"找到{mode}模式的结果: {len(dirs)}个")
            for dir_path in dirs[:3]:  # 只显示前3个
                print(f"  - {dir_path}")
            if len(dirs) > 3:
                print(f"  ... 以及{len(dirs)-3}个更多结果")
    
    if not found_results:
        print(f"错误：在{args.logs_dir}中未找到任何结果目录")
        return
    
    # 比较结果
    compare_results(dirs_dict, args.output_dir)

if __name__ == "__main__":
    main() 