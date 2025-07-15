#!/usr/bin/env python3
"""
HARL-based Innovation 1 Validation Runner
运行基于HARL框架的创新点一验证
"""

import os
import sys
import subprocess
import argparse

def setup_environment():
    """设置环境变量和路径"""
    # 添加当前目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 添加HARL路径
    harl_path = os.path.join(current_dir, 'harl')
    if harl_path not in sys.path:
        sys.path.insert(0, harl_path)
    
    print(f"✓ Current directory: {current_dir}")
    print(f"✓ HARL path: {harl_path}")
    print(f"✓ Python path configured")

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'torch',
        'numpy',
        'matplotlib',
        'tensorboard',
        'yaml',
        'gym'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} found")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    if missing_packages:
        print(f"\n警告: 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_validation(config_path=None):
    """运行验证"""
    try:
        # 设置环境
        setup_environment()
        
        # 检查依赖
        if not check_dependencies():
            return False
        
        # 导入验证器
        from harl_based_innovation1_validation import HARLBasedInnovation1Validator
        
        # 创建验证器
        if config_path:
            validator = HARLBasedInnovation1Validator(config_path)
        else:
            validator = HARLBasedInnovation1Validator()
        
        # 运行验证
        validator.run_validation()
        
        return True
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有HARL框架文件都在正确位置")
        return False
    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HARL-based Innovation 1 Validation')
    parser.add_argument('--config', type=str, default='harl_innovation1_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check dependencies and setup')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HARL-based Innovation 1 Validation Runner")
    print("基于HARL框架的创新点一验证启动器")
    print("="*60)
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装缺少的依赖包")
        return
    
    if args.check_only:
        print("\n✓ 环境检查完成，所有依赖都已满足")
        return
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"\n警告: 配置文件 {args.config} 不存在")
        print("将使用默认配置")
    
    # 运行验证
    print(f"\n开始运行验证...")
    print(f"配置文件: {args.config}")
    
    success = run_validation(args.config)
    
    if success:
        print("\n✓ 验证成功完成！")
        print("请查看 logs/harl_innovation1_validation 目录获取结果")
    else:
        print("\n✗ 验证失败")

if __name__ == "__main__":
    main() 