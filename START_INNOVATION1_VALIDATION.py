#!/usr/bin/env python3
"""
创新点1验证快速启动脚本
Quick Start Script for Innovation 1 Validation

使用方法：
1. 直接运行：python START_INNOVATION1_VALIDATION.py
2. 或者使用参数：python START_INNOVATION1_VALIDATION.py --config custom_config.yaml

该脚本将：
1. 自动检查环境依赖
2. 验证HARL框架组件
3. 运行创新点1验证实验
4. 生成验证报告
"""

import os
import sys
import subprocess
import argparse
import platform

def print_banner():
    """打印启动横幅"""
    print("="*80)
    print("🚀 HARL-Based Innovation 1 Validation")
    print("   动态上下文感知状态表征验证 (Dynamic Context-Aware State Representation)")
    print("="*80)
    print()

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"🐍 Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ 错误: 需要Python 3.7或更高版本")
        return False
    
    print("✅ Python版本检查通过")
    return True

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('tensorboard', 'TensorBoard'),
        ('yaml', 'PyYAML'),
        ('gym', 'OpenAI Gym'),
        ('tqdm', 'tqdm'),
        ('scipy', 'SciPy'),
        ('seaborn', 'Seaborn'),
        ('pandas', 'Pandas')
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {name} (缺失)")
    
    if missing_packages:
        print(f"\n⚠️  缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True

def check_harl_framework():
    """检查HARL框架"""
    print("\n🧠 检查HARL框架...")
    
    required_files = [
        ('harl/algorithms/actors/hasac.py', 'HASAC算法'),
        ('harl/models/policy_models/transformer_policy.py', 'Transformer策略'),
        ('harl/utils/contrastive_learning.py', '对比学习模块'),
        ('harl/models/base/transformer.py', 'Transformer基础模块'),
        ('harl/algorithms/critics/soft_twin_continuous_q_critic.py', '软双Q批评家'),
        ('harl/common/buffers/off_policy_buffer_ep.py', '离策略缓冲区'),
        ('harl/utils/configs_tools.py', '配置工具')
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {description}")
        else:
            missing_files.append((file_path, description))
            print(f"❌ {description} ({file_path})")
    
    if missing_files:
        print(f"\n⚠️  缺少以下HARL框架文件:")
        for file_path, description in missing_files:
            print(f"   - {description}: {file_path}")
        return False
    
    print("✅ HARL框架检查通过")
    return True

def check_validation_files():
    """检查验证文件"""
    print("\n📋 检查验证文件...")
    
    required_files = [
        ('harl_based_innovation1_validation.py', '创新点1验证主程序'),
        ('run_harl_innovation1_validation.py', '验证运行器'),
        ('harl_innovation1_config.yaml', '验证配置文件'),
        ('hasac_flow_mec_v2x_env.py', 'MEC-V2X环境'),
        ('complete_mec_v2x_simulation.py', '完整MEC-V2X仿真'),
        ('requirements.txt', '依赖需求文件')
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {description}")
        else:
            missing_files.append((file_path, description))
            print(f"❌ {description} ({file_path})")
    
    if missing_files:
        print(f"\n⚠️  缺少以下验证文件:")
        for file_path, description in missing_files:
            print(f"   - {description}: {file_path}")
        return False
    
    print("✅ 验证文件检查通过")
    return True

def setup_environment():
    """设置环境"""
    print("\n🔧 设置环境...")
    
    # 添加当前目录到Python路径
    current_dir = os.path.abspath('.')
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 添加HARL路径
    harl_path = os.path.join(current_dir, 'harl')
    if harl_path not in sys.path:
        sys.path.insert(0, harl_path)
    
    print(f"✅ 当前目录: {current_dir}")
    print(f"✅ HARL路径: {harl_path}")
    print("✅ 环境设置完成")
    return True

def run_validation(config_path=None):
    """运行验证"""
    print("\n🚀 启动创新点1验证...")
    
    try:
        # 导入验证模块
        from harl_based_innovation1_validation import HARLBasedInnovation1Validator
        
        # 创建验证器
        config_file = config_path or "harl_innovation1_config.yaml"
        validator = HARLBasedInnovation1Validator(config_file)
        
        print(f"✅ 使用配置文件: {config_file}")
        print("🔄 开始验证...")
        
        # 运行验证
        results = validator.run_validation()
        
        print("✅ 验证完成!")
        if results and isinstance(results, dict):
            print(f"📊 结果保存在: {results.get('log_dir', 'logs/')}")
        else:
            print("📊 结果保存在: logs/")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        print("请检查错误信息并重试")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创新点1验证快速启动')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--skip-checks', action='store_true', help='跳过环境检查')
    args = parser.parse_args()
    
    print_banner()
    
    # 环境检查
    if not args.skip_checks:
        checks = [
            check_python_version(),
            check_dependencies(),
            check_harl_framework(),
            check_validation_files(),
            setup_environment()
        ]
        
        if not all(checks):
            print("\n❌ 环境检查失败，请解决上述问题后重试")
            return False
    
    # 运行验证
    success = run_validation(args.config)
    
    if success:
        print("\n🎉 创新点1验证成功完成!")
        print("📊 请查看生成的报告和图表")
    else:
        print("\n❌ 验证失败")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 