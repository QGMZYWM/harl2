# 手动安装tqdm库指南

## 🔧 如果您遇到网络问题，可以尝试以下方法安装tqdm：

### 方法1：使用conda安装（推荐）
```bash
conda install tqdm
```

### 方法2：使用不同的PyPI镜像
```bash
# 使用阿里云镜像
pip install -i https://mirrors.aliyun.com/pypi/simple/ tqdm

# 使用中科大镜像
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ tqdm

# 使用豆瓣镜像
pip install -i https://pypi.douban.com/simple/ tqdm
```

### 方法3：离线安装
如果网络完全不可用，可以：
1. 在有网络的机器上下载tqdm的whl文件
2. 传输到目标机器
3. 使用以下命令安装：
```bash
pip install tqdm-4.65.0-py3-none-any.whl
```

### 方法4：使用系统包管理器
```bash
# Ubuntu/Debian
sudo apt-get install python3-tqdm

# CentOS/RHEL
sudo yum install python3-tqdm
```

## 📝 注意事项

- 如果以上方法都不行，**程序会自动使用简单的进度显示**，功能完全正常
- 简单进度显示包含：
  - 百分比进度
  - 进度条
  - 剩余时间估算
  - 实时状态更新

## 🚀 运行实验

无论是否安装了tqdm，都可以正常运行：
```bash
python real_v2x_experiment.py
```

程序会自动检测tqdm是否可用，并选择最佳的显示方式。 