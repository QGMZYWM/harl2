#!/bin/bash

# HARL完整环境安装脚本
# 适配CUDA 10.1版本

echo "🚀 开始安装HARL环境..."
echo "检测到CUDA 10.1版本"

# 激活conda环境
echo "📦 激活虚拟环境..."
source activate /home/stu16/.conda/envs/harl

# 第1步：安装PyTorch (CUDA 10.1兼容版本)
echo "🔥 安装PyTorch (CUDA 10.1)..."
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.1 -c pytorch -y

# 第2步：安装基础科学计算包
echo "🧮 安装科学计算依赖..."
conda install numpy>=1.19.0 matplotlib>=3.3.0 scipy>=1.5.0 pandas>=1.1.0 -y

# 第3步：安装其他依赖
echo "📚 安装其他依赖..."
pip install --user pyyaml>=5.3.1
pip install --user tensorboard>=2.2.1
pip install --user tensorboardX
pip install --user setproctitle
pip install --user seaborn>=0.11.0
pip install --user gym==0.21.0
pip install --user pyglet==1.5.0
pip install --user importlib-metadata==4.13.0
pip install --user networkx>=2.5

# 第4步：安装HARL框架
echo "🎯 安装HARL框架..."
cd /home/stu16/HARL-main
pip install --user -e .

# 第5步：验证安装
echo "✅ 验证安装..."
python test_installation.py

echo "🎉 安装完成！"
echo "现在可以运行: python simple_real_test.py" 