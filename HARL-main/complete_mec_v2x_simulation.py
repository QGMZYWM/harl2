#!/usr/bin/env python3
"""
完整的MEC-V2X仿真实验环境

实现真实的移动边缘计算与车联网融合仿真，包括：
1. 详细的MEC服务器建模
2. 完整的任务卸载机制  
3. 真实的V2X通信协议
4. 精确的性能评估指标
"""

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
import random
import math
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time

# 任务类型枚举
class TaskType(Enum):
    COMPUTE_INTENSIVE = "compute_intensive"    # 计算密集型
    DATA_INTENSIVE = "data_intensive"          # 数据密集型
    LATENCY_SENSITIVE = "latency_sensitive"    # 延迟敏感型
    MIXED = "mixed"                           # 混合型

# 卸载策略枚举
class OffloadingStrategy(Enum):
    LOCAL = "local"           # 本地执行
    RSU_MEC = "rsu_mec"      # RSU-MEC卸载
    V2V = "v2v"              # 车车协作
    CLOUD = "cloud"          # 云端卸载
    HYBRID = "hybrid"        # 混合策略

@dataclass
class Task:
    """任务模型"""
    task_id: str
    task_type: TaskType
    cpu_cycles: float          # 所需CPU周期 (GHz*s)
    data_size: float           # 数据大小 (MB)
    deadline: float            # 截止时间 (s)
    priority: float            # 优先级 [0,1]
    arrival_time: float        # 到达时间
    dependency: List[str]      # 依赖任务列表
    sensitivity: float         # 数据敏感度 [0,1]
    
@dataclass
class MECServer:
    """MEC服务器模型"""
    server_id: str
    cpu_capacity: float        # CPU容量 (GHz)
    memory_capacity: float     # 内存容量 (GB)
    storage_capacity: float    # 存储容量 (GB)
    current_cpu_load: float    # 当前CPU负载
    current_memory_load: float # 当前内存负载
    task_queue: List[Task]     # 任务队列
    processing_tasks: List[Task] # 正在处理的任务
    
@dataclass
class Vehicle:
    """车辆模型"""
    vehicle_id: str
    position: np.ndarray       # 位置 (x, y)
    velocity: np.ndarray       # 速度 (vx, vy)
    acceleration: np.ndarray   # 加速度 (ax, ay)
    cpu_capacity: float        # CPU容量 (GHz)
    memory_capacity: float     # 内存容量 (GB)
    battery_level: float       # 电池电量 [0,1]
    thermal_state: float       # 热状态 [0,1]
    current_tasks: List[Task]  # 当前任务列表
    offloading_history: deque  # 卸载历史
    trust_score: float         # 信任评分 [0,1]
    communication_power: float # 通信功率 (W)
    
@dataclass
class RSU:
    """路边单元模型"""
    rsu_id: str
    position: np.ndarray       # 位置 (x, y)
    coverage_radius: float     # 覆盖半径 (m)
    mec_server: MECServer      # 关联的MEC服务器
    connected_vehicles: List[str] # 连接的车辆列表
    channel_bandwidth: float   # 信道带宽 (MHz)
    transmission_power: float  # 传输功率 (W)

class V2XCommunicationModel:
    """V2X通信模型"""
    
    def __init__(self):
        # 通信参数
        self.frequency = 5.9e9          # 载波频率 (Hz) - DSRC
        self.light_speed = 3e8          # 光速 (m/s)
        self.noise_power_dbm = -110     # 噪声功率 (dBm)
        self.path_loss_exponent = 2.5   # 路径损耗指数
        self.shadowing_std = 8          # 阴影衰落标准差 (dB)
        
    def calculate_path_loss(self, distance: float) -> float:
        """计算路径损耗 (dB)"""
        if distance < 1:
            distance = 1
        wavelength = self.light_speed / self.frequency
        path_loss_db = 20 * math.log10(4 * math.pi * distance / wavelength)
        path_loss_db += 10 * self.path_loss_exponent * math.log10(distance)
        return path_loss_db
    
    def calculate_channel_gain(self, distance: float) -> float:
        """计算信道增益"""
        path_loss_db = self.calculate_path_loss(distance)
        shadowing_db = np.random.normal(0, self.shadowing_std)
        total_loss_db = path_loss_db + shadowing_db
        channel_gain = 10 ** (-total_loss_db / 10)
        return channel_gain
    
    def calculate_snr(self, tx_power_w: float, distance: float) -> float:
        """计算信噪比 (dB)"""
        channel_gain = self.calculate_channel_gain(distance)
        received_power_w = tx_power_w * channel_gain
        received_power_dbm = 10 * math.log10(received_power_w * 1000)
        snr_db = received_power_dbm - self.noise_power_dbm
        return snr_db
    
    def calculate_data_rate(self, snr_db: float, bandwidth_hz: float) -> float:
        """计算数据传输速率 (bps)"""
        snr_linear = 10 ** (snr_db / 10)
        data_rate = bandwidth_hz * math.log2(1 + snr_linear)
        return data_rate
    
    def calculate_transmission_time(self, data_size_mb: float, data_rate_bps: float) -> float:
        """计算传输时间 (s)"""
        data_size_bits = data_size_mb * 8 * 1e6
        transmission_time = data_size_bits / data_rate_bps if data_rate_bps > 0 else float('inf')
        return transmission_time

class TaskGenerator:
    """任务生成器"""
    
    def __init__(self):
        self.task_counter = 0
        
    def generate_task(self, current_time: float) -> Task:
        """生成随机任务"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        # 随机选择任务类型
        task_type = random.choice(list(TaskType))
        
        # 根据任务类型设置参数
        if task_type == TaskType.COMPUTE_INTENSIVE:
            cpu_cycles = np.random.uniform(5.0, 20.0)      # 高计算需求
            data_size = np.random.uniform(1.0, 10.0)       # 低数据传输
            deadline = np.random.uniform(10.0, 30.0)       # 中等延迟要求
        elif task_type == TaskType.DATA_INTENSIVE:
            cpu_cycles = np.random.uniform(1.0, 5.0)       # 低计算需求
            data_size = np.random.uniform(50.0, 200.0)     # 高数据传输
            deadline = np.random.uniform(20.0, 60.0)       # 宽松延迟要求
        elif task_type == TaskType.LATENCY_SENSITIVE:
            cpu_cycles = np.random.uniform(0.5, 3.0)       # 低计算需求
            data_size = np.random.uniform(0.1, 5.0)        # 低数据传输
            deadline = np.random.uniform(1.0, 5.0)         # 严格延迟要求
        else:  # MIXED
            cpu_cycles = np.random.uniform(3.0, 15.0)      # 中等计算需求
            data_size = np.random.uniform(10.0, 50.0)      # 中等数据传输
            deadline = np.random.uniform(5.0, 25.0)        # 中等延迟要求
        
        return Task(
            task_id=task_id,
            task_type=task_type,
            cpu_cycles=cpu_cycles,
            data_size=data_size,
            deadline=deadline,
            priority=np.random.uniform(0.3, 1.0),
            arrival_time=current_time,
            dependency=[],
            sensitivity=np.random.uniform(0.0, 1.0)
        )

class CompleteMECV2XSimulation:
    """完整的MEC-V2X仿真环境"""
    
    def __init__(self, config: Dict):
        """
        初始化仿真环境
        
        Args:
            config: 仿真配置参数
        """
        self.config = config
        self.current_time = 0.0
        self.time_step = config.get('time_step', 0.1)  # 仿真时间步长 (s)
        
        # 初始化组件
        self.vehicles = self._initialize_vehicles()
        self.rsus = self._initialize_rsus()
        self.communication_model = V2XCommunicationModel()
        self.task_generator = TaskGenerator()
        
        # 性能指标
        self.metrics = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_latency': 0.0,
            'total_energy_consumption': 0.0,
            'mec_utilization': [],
            'v2v_collaborations': 0
        }
        
        # 动作和观测空间
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )  # [local_ratio, rsu_ratio, v2v_ratio, cloud_ratio, tx_power]
        
        obs_dim = 50  # 复杂的状态表示
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        print(f"✓ Complete MEC-V2X Simulation Environment Initialized")
        print(f"  Vehicles: {len(self.vehicles)}")
        print(f"  RSUs: {len(self.rsus)}")
        print(f"  Time Step: {self.time_step}s")
    
    def _initialize_vehicles(self) -> List[Vehicle]:
        """初始化车辆"""
        vehicles = []
        num_vehicles = self.config.get('num_vehicles', 10)
        map_size = self.config.get('map_size', 1000.0)
        
        for i in range(num_vehicles):
            vehicle = Vehicle(
                vehicle_id=f"vehicle_{i}",
                position=np.random.uniform(0, map_size, 2),
                velocity=np.random.uniform(-20, 20, 2),  # m/s
                acceleration=np.zeros(2),
                cpu_capacity=np.random.uniform(1.0, 8.0),  # GHz
                memory_capacity=np.random.uniform(2.0, 16.0),  # GB
                battery_level=np.random.uniform(0.5, 1.0),
                thermal_state=np.random.uniform(0.2, 0.6),
                current_tasks=[],
                offloading_history=deque(maxlen=100),
                trust_score=np.random.uniform(0.7, 1.0),
                communication_power=np.random.uniform(0.1, 2.0)  # W
            )
            vehicles.append(vehicle)
        
        return vehicles
    
    def _initialize_rsus(self) -> List[RSU]:
        """初始化RSU和MEC服务器"""
        rsus = []
        num_rsus = self.config.get('num_rsus', 4)
        map_size = self.config.get('map_size', 1000.0)
        
        for i in range(num_rsus):
            # 创建MEC服务器
            mec_server = MECServer(
                server_id=f"mec_{i}",
                cpu_capacity=np.random.uniform(50.0, 200.0),  # GHz
                memory_capacity=np.random.uniform(64.0, 512.0),  # GB
                storage_capacity=np.random.uniform(1000.0, 10000.0),  # GB
                current_cpu_load=np.random.uniform(0.1, 0.3),
                current_memory_load=np.random.uniform(0.1, 0.3),
                task_queue=[],
                processing_tasks=[]
            )
            
            # 创建RSU
            rsu = RSU(
                rsu_id=f"rsu_{i}",
                position=np.array([
                    (i + 0.5) * map_size / num_rsus + np.random.uniform(-50, 50),
                    map_size / 2 + np.random.uniform(-100, 100)
                ]),
                coverage_radius=np.random.uniform(200.0, 500.0),  # m
                mec_server=mec_server,
                connected_vehicles=[],
                channel_bandwidth=20e6,  # 20 MHz
                transmission_power=10.0   # W
            )
            rsus.append(rsu)
        
        return rsus
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置仿真环境"""
        self.current_time = 0.0
        
        # 重置车辆状态
        for vehicle in self.vehicles:
            vehicle.current_tasks = []
            vehicle.battery_level = np.random.uniform(0.5, 1.0)
            vehicle.thermal_state = np.random.uniform(0.2, 0.6)
            vehicle.offloading_history.clear()
        
        # 重置MEC服务器
        for rsu in self.rsus:
            rsu.mec_server.task_queue = []
            rsu.mec_server.processing_tasks = []
            rsu.mec_server.current_cpu_load = np.random.uniform(0.1, 0.3)
            rsu.connected_vehicles = []
        
        # 重置指标
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            else:
                self.metrics[key] = 0
        
        # 返回初始观测
        observations = {}
        for i, vehicle in enumerate(self.vehicles):
            observations[f'vehicle_{i}'] = self._get_vehicle_observation(vehicle)
        
        return observations
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict]:
        """执行一个仿真步骤"""
        # 1. 生成新任务
        self._generate_tasks()
        
        # 2. 更新车辆动态
        self._update_vehicle_dynamics()
        
        # 3. 更新RSU连接
        self._update_rsu_connections()
        
        # 4. 执行卸载决策
        rewards = self._execute_offloading_decisions(actions)
        
        # 5. 处理任务
        self._process_tasks()
        
        # 6. 更新MEC服务器状态
        self._update_mec_servers()
        
        # 7. 计算奖励和观测
        observations = {}
        dones = {}
        
        for i, vehicle in enumerate(self.vehicles):
            observations[f'vehicle_{i}'] = self._get_vehicle_observation(vehicle)
            dones[f'vehicle_{i}'] = False  # 连续仿真
        
        # 8. 更新时间
        self.current_time += self.time_step
        
        # 9. 收集仿真信息
        info = self._collect_simulation_info()
        
        return observations, rewards, dones, info
    
    def _generate_tasks(self):
        """生成新任务"""
        task_generation_prob = self.config.get('task_generation_prob', 0.3)
        
        for vehicle in self.vehicles:
            if np.random.random() < task_generation_prob:
                new_task = self.task_generator.generate_task(self.current_time)
                vehicle.current_tasks.append(new_task)
    
    def _update_vehicle_dynamics(self):
        """更新车辆运动动态"""
        for vehicle in self.vehicles:
            # 简单的车辆运动模型
            # 随机加速度变化
            vehicle.acceleration = np.random.normal(0, 0.5, 2)
            vehicle.acceleration = np.clip(vehicle.acceleration, -3.0, 3.0)
            
            # 更新速度
            vehicle.velocity += vehicle.acceleration * self.time_step
            vehicle.velocity = np.clip(vehicle.velocity, -30, 30)  # 限速
            
            # 更新位置
            vehicle.position += vehicle.velocity * self.time_step
            
            # 边界处理（环绕）
            map_size = self.config.get('map_size', 1000.0)
            vehicle.position = vehicle.position % map_size
    
    def _update_rsu_connections(self):
        """更新RSU连接状态"""
        for rsu in self.rsus:
            rsu.connected_vehicles = []
            
            for vehicle in self.vehicles:
                distance = np.linalg.norm(vehicle.position - rsu.position)
                if distance <= rsu.coverage_radius:
                    rsu.connected_vehicles.append(vehicle.vehicle_id)
    
    def _execute_offloading_decisions(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """执行卸载决策并计算奖励"""
        rewards = {}
        
        for i, vehicle in enumerate(self.vehicles):
            vehicle_id = f'vehicle_{i}'
            action = actions.get(vehicle_id, np.array([0.7, 0.1, 0.1, 0.1, 0.5]))
            
            # 解析动作
            local_ratio = action[0]
            rsu_ratio = action[1]
            v2v_ratio = action[2]
            cloud_ratio = action[3]
            tx_power = action[4]
            
            # 归一化策略比例
            total_ratio = local_ratio + rsu_ratio + v2v_ratio + cloud_ratio
            if total_ratio > 0:
                local_ratio /= total_ratio
                rsu_ratio /= total_ratio
                v2v_ratio /= total_ratio
                cloud_ratio /= total_ratio
            
            # 执行卸载并计算奖励
            reward = self._offload_vehicle_tasks(
                vehicle, local_ratio, rsu_ratio, v2v_ratio, cloud_ratio, tx_power
            )
            rewards[vehicle_id] = reward
        
        return rewards
    
    def _offload_vehicle_tasks(self, vehicle: Vehicle, local_ratio: float, 
                              rsu_ratio: float, v2v_ratio: float, 
                              cloud_ratio: float, tx_power: float) -> float:
        """执行特定车辆的任务卸载"""
        if not vehicle.current_tasks:
            return 0.0
        
        total_reward = 0.0
        tasks_to_remove = []
        
        for task in vehicle.current_tasks:
            # 决定卸载策略
            strategy = self._determine_offloading_strategy(
                task, local_ratio, rsu_ratio, v2v_ratio, cloud_ratio
            )
            
            # 执行卸载
            success, latency, energy = self._execute_task_offloading(
                vehicle, task, strategy, tx_power
            )
            
            if success:
                # 任务成功完成
                self.metrics['completed_tasks'] += 1
                self.metrics['total_latency'] += latency
                self.metrics['total_energy_consumption'] += energy
                
                # 计算任务奖励
                task_reward = self._calculate_task_reward(task, latency, energy, success)
                total_reward += task_reward
                
                tasks_to_remove.append(task)
            else:
                # 任务失败
                deadline_violation = (self.current_time - task.arrival_time) > task.deadline
                if deadline_violation:
                    self.metrics['failed_tasks'] += 1
                    total_reward -= 10.0  # 失败惩罚
                    tasks_to_remove.append(task)
        
        # 移除已完成/失败的任务
        for task in tasks_to_remove:
            vehicle.current_tasks.remove(task)
        
        # 添加系统级奖励
        system_reward = self._calculate_system_reward(vehicle)
        total_reward += system_reward
        
        return total_reward
    
    def _determine_offloading_strategy(self, task: Task, local_ratio: float,
                                     rsu_ratio: float, v2v_ratio: float,
                                     cloud_ratio: float) -> OffloadingStrategy:
        """确定卸载策略"""
        strategies = [
            (OffloadingStrategy.LOCAL, local_ratio),
            (OffloadingStrategy.RSU_MEC, rsu_ratio),
            (OffloadingStrategy.V2V, v2v_ratio),
            (OffloadingStrategy.CLOUD, cloud_ratio)
        ]
        
        # 根据概率选择策略
        total_prob = sum(prob for _, prob in strategies)
        if total_prob == 0:
            return OffloadingStrategy.LOCAL
        
        rand_val = np.random.random() * total_prob
        cumulative_prob = 0
        
        for strategy, prob in strategies:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return strategy
        
        return OffloadingStrategy.LOCAL
    
    def _execute_task_offloading(self, vehicle: Vehicle, task: Task, 
                                strategy: OffloadingStrategy, tx_power: float) -> Tuple[bool, float, float]:
        """执行具体的任务卸载"""
        if strategy == OffloadingStrategy.LOCAL:
            return self._execute_local_processing(vehicle, task)
        elif strategy == OffloadingStrategy.RSU_MEC:
            return self._execute_rsu_offloading(vehicle, task, tx_power)
        elif strategy == OffloadingStrategy.V2V:
            return self._execute_v2v_offloading(vehicle, task, tx_power)
        elif strategy == OffloadingStrategy.CLOUD:
            return self._execute_cloud_offloading(vehicle, task, tx_power)
        else:
            return self._execute_local_processing(vehicle, task)
    
    def _execute_local_processing(self, vehicle: Vehicle, task: Task) -> Tuple[bool, float, float]:
        """本地处理"""
        # 计算处理时间
        processing_time = task.cpu_cycles / vehicle.cpu_capacity
        
        # 计算能耗 (简化模型)
        energy_consumption = task.cpu_cycles * 0.5  # J/GHz*s
        
        # 检查截止时间
        total_time = processing_time
        deadline_met = total_time <= task.deadline
        
        # 更新车辆状态
        vehicle.battery_level -= energy_consumption / 10000  # 简化电池模型
        vehicle.thermal_state += processing_time * 0.01      # 热量积累
        vehicle.thermal_state = min(vehicle.thermal_state, 1.0)
        
        return deadline_met, total_time, energy_consumption
    
    def _execute_rsu_offloading(self, vehicle: Vehicle, task: Task, tx_power: float) -> Tuple[bool, float, float]:
        """RSU-MEC卸载"""
        # 找到最近的可用RSU
        best_rsu = None
        min_distance = float('inf')
        
        for rsu in self.rsus:
            distance = np.linalg.norm(vehicle.position - rsu.position)
            if distance <= rsu.coverage_radius and distance < min_distance:
                min_distance = distance
                best_rsu = rsu
        
        if best_rsu is None:
            # 没有可用RSU，回退到本地处理
            return self._execute_local_processing(vehicle, task)
        
        # 计算通信参数
        snr_db = self.communication_model.calculate_snr(tx_power, min_distance)
        data_rate = self.communication_model.calculate_data_rate(snr_db, best_rsu.channel_bandwidth)
        
        # 计算传输时间
        upload_time = self.communication_model.calculate_transmission_time(task.data_size, data_rate)
        download_time = self.communication_model.calculate_transmission_time(task.data_size * 0.1, data_rate)  # 结果较小
        
        # 计算MEC处理时间
        mec_processing_time = task.cpu_cycles / best_rsu.mec_server.cpu_capacity
        
        # 计算总时间和能耗
        total_time = upload_time + mec_processing_time + download_time
        communication_energy = tx_power * (upload_time + download_time)
        
        # 检查截止时间
        deadline_met = total_time <= task.deadline
        
        if deadline_met:
            # 更新MEC服务器负载
            best_rsu.mec_server.current_cpu_load += task.cpu_cycles / best_rsu.mec_server.cpu_capacity * 0.1
            best_rsu.mec_server.current_cpu_load = min(best_rsu.mec_server.current_cpu_load, 1.0)
        
        # 更新车辆状态
        vehicle.battery_level -= communication_energy / 10000
        
        return deadline_met, total_time, communication_energy
    
    def _execute_v2v_offloading(self, vehicle: Vehicle, task: Task, tx_power: float) -> Tuple[bool, float, float]:
        """V2V协作卸载"""
        # 寻找合适的协作车辆
        suitable_vehicles = []
        
        for other_vehicle in self.vehicles:
            if other_vehicle.vehicle_id == vehicle.vehicle_id:
                continue
            
            distance = np.linalg.norm(vehicle.position - other_vehicle.position)
            if (distance <= 300 and  # 通信范围
                other_vehicle.cpu_capacity > vehicle.cpu_capacity * 1.2 and  # 计算能力更强
                other_vehicle.battery_level > 0.5 and  # 电量充足
                len(other_vehicle.current_tasks) < 3):  # 负载不重
                suitable_vehicles.append((other_vehicle, distance))
        
        if not suitable_vehicles:
            # 没有合适的协作车辆，回退到本地处理
            return self._execute_local_processing(vehicle, task)
        
        # 选择最近的合适车辆
        best_vehicle, min_distance = min(suitable_vehicles, key=lambda x: x[1])
        
        # 计算V2V通信参数
        snr_db = self.communication_model.calculate_snr(tx_power, min_distance)
        data_rate = self.communication_model.calculate_data_rate(snr_db, 10e6)  # 10 MHz V2V
        
        # 计算时间和能耗
        upload_time = self.communication_model.calculate_transmission_time(task.data_size, data_rate)
        download_time = self.communication_model.calculate_transmission_time(task.data_size * 0.1, data_rate)
        processing_time = task.cpu_cycles / best_vehicle.cpu_capacity
        
        total_time = upload_time + processing_time + download_time
        communication_energy = tx_power * (upload_time + download_time)
        
        deadline_met = total_time <= task.deadline
        
        if deadline_met:
            self.metrics['v2v_collaborations'] += 1
            # 更新协作车辆状态
            best_vehicle.battery_level -= task.cpu_cycles * 0.3 / 10000
        
        return deadline_met, total_time, communication_energy
    
    def _execute_cloud_offloading(self, vehicle: Vehicle, task: Task, tx_power: float) -> Tuple[bool, float, float]:
        """云端卸载"""
        # 云端有无限计算资源，但有较高的传输延迟
        base_latency = 50e-3  # 50ms 基础延迟
        
        # 计算数据传输时间（简化模型）
        upload_rate = 50e6   # 50 Mbps 上传速率
        download_rate = 100e6  # 100 Mbps 下载速率
        
        upload_time = task.data_size * 8 / upload_rate
        download_time = task.data_size * 0.1 * 8 / download_rate
        processing_time = task.cpu_cycles / 1000  # 云端有强大计算能力
        
        total_time = base_latency + upload_time + processing_time + download_time
        communication_energy = 2.0 * (upload_time + download_time)  # 较高功耗
        
        deadline_met = total_time <= task.deadline
        
        return deadline_met, total_time, communication_energy
    
    def _calculate_task_reward(self, task: Task, latency: float, energy: float, success: bool) -> float:
        """计算任务奖励"""
        if not success:
            return -5.0
        
        # 基础完成奖励
        base_reward = 10.0 * task.priority
        
        # 延迟奖励（越低越好）
        latency_reward = max(0, (task.deadline - latency) / task.deadline) * 5.0
        
        # 能耗奖励（越低越好）
        energy_reward = max(0, (10.0 - energy) / 10.0) * 3.0
        
        return base_reward + latency_reward + energy_reward
    
    def _calculate_system_reward(self, vehicle: Vehicle) -> float:
        """计算系统级奖励"""
        # 电池水平奖励
        battery_reward = vehicle.battery_level * 2.0
        
        # 热状态奖励（越低越好）
        thermal_reward = (1.0 - vehicle.thermal_state) * 1.0
        
        # 负载均衡奖励
        task_load = len(vehicle.current_tasks)
        load_reward = max(0, (5 - task_load) / 5) * 1.0
        
        return battery_reward + thermal_reward + load_reward
    
    def _process_tasks(self):
        """处理正在执行的任务"""
        # 更新MEC服务器任务队列
        for rsu in self.rsus:
            mec = rsu.mec_server
            completed_tasks = []
            
            for task in mec.processing_tasks:
                # 简化的任务完成检查
                if np.random.random() < 0.1:  # 10%概率完成
                    completed_tasks.append(task)
            
            for task in completed_tasks:
                mec.processing_tasks.remove(task)
                mec.current_cpu_load -= 0.05
                mec.current_cpu_load = max(mec.current_cpu_load, 0.0)
    
    def _update_mec_servers(self):
        """更新MEC服务器状态"""
        utilizations = []
        
        for rsu in self.rsus:
            mec = rsu.mec_server
            utilization = mec.current_cpu_load
            utilizations.append(utilization)
            
            # 负载自然衰减
            mec.current_cpu_load *= 0.95
            mec.current_memory_load *= 0.95
        
        self.metrics['mec_utilization'].append(np.mean(utilizations))
    
    def _get_vehicle_observation(self, vehicle: Vehicle) -> np.ndarray:
        """获取车辆观测"""
        obs = []
        
        # 车辆基础状态 (10维)
        obs.extend([
            vehicle.position[0] / self.config.get('map_size', 1000),  # 归一化位置x
            vehicle.position[1] / self.config.get('map_size', 1000),  # 归一化位置y
            vehicle.velocity[0] / 30.0,                               # 归一化速度x
            vehicle.velocity[1] / 30.0,                               # 归一化速度y
            vehicle.cpu_capacity / 8.0,                              # 归一化CPU容量
            vehicle.memory_capacity / 16.0,                          # 归一化内存容量
            vehicle.battery_level,                                   # 电池电量
            vehicle.thermal_state,                                   # 热状态
            len(vehicle.current_tasks) / 10.0,                       # 归一化任务数量
            vehicle.trust_score                                      # 信任评分
        ])
        
        # RSU信息 (20维 = 5个RSU * 4维)
        rsu_info = []
        for rsu in self.rsus[:5]:  # 最多5个RSU
            distance = np.linalg.norm(vehicle.position - rsu.position)
            in_coverage = 1.0 if distance <= rsu.coverage_radius else 0.0
            rsu_info.extend([
                distance / 1000.0,                                   # 归一化距离
                in_coverage,                                         # 覆盖状态
                rsu.mec_server.current_cpu_load,                     # MEC负载
                len(rsu.connected_vehicles) / 10.0                   # 连接车辆数量
            ])
        
        # 如果RSU少于5个，用0填充
        while len(rsu_info) < 20:
            rsu_info.extend([0.0, 0.0, 0.0, 0.0])
        
        obs.extend(rsu_info)
        
        # 任务信息 (20维 = 当前任务的统计信息)
        if vehicle.current_tasks:
            task_types = [task.task_type.value for task in vehicle.current_tasks]
            avg_cpu_cycles = np.mean([task.cpu_cycles for task in vehicle.current_tasks])
            avg_data_size = np.mean([task.data_size for task in vehicle.current_tasks])
            avg_deadline = np.mean([task.deadline for task in vehicle.current_tasks])
            avg_priority = np.mean([task.priority for task in vehicle.current_tasks])
            
            # 任务类型分布
            type_dist = [0, 0, 0, 0]  # 四种任务类型
            for task_type in task_types:
                if task_type == TaskType.COMPUTE_INTENSIVE.value:
                    type_dist[0] += 1
                elif task_type == TaskType.DATA_INTENSIVE.value:
                    type_dist[1] += 1
                elif task_type == TaskType.LATENCY_SENSITIVE.value:
                    type_dist[2] += 1
                else:
                    type_dist[3] += 1
            
            type_dist = [x / len(vehicle.current_tasks) for x in type_dist]
            
            task_info = [
                len(vehicle.current_tasks) / 10.0,     # 任务数量
                avg_cpu_cycles / 20.0,                 # 平均CPU需求
                avg_data_size / 200.0,                 # 平均数据大小
                avg_deadline / 60.0,                   # 平均截止时间
                avg_priority,                          # 平均优先级
            ]
            task_info.extend(type_dist)              # 任务类型分布 (4维)
            
            # 填充剩余维度
            while len(task_info) < 20:
                task_info.append(0.0)
        else:
            task_info = [0.0] * 20
        
        obs.extend(task_info)
        
        # 确保观测维度正确
        while len(obs) < 50:
            obs.append(0.0)
        
        return np.array(obs[:50], dtype=np.float32)
    
    def _collect_simulation_info(self) -> Dict:
        """收集仿真信息"""
        info = {
            'current_time': self.current_time,
            'metrics': self.metrics.copy(),
            'vehicle_states': [],
            'rsu_states': [],
            'communication_stats': {
                'active_connections': sum(len(rsu.connected_vehicles) for rsu in self.rsus),
                'average_mec_utilization': np.mean(self.metrics['mec_utilization']) if self.metrics['mec_utilization'] else 0.0
            }
        }
        
        # 车辆状态信息
        for vehicle in self.vehicles:
            vehicle_state = {
                'vehicle_id': vehicle.vehicle_id,
                'position': vehicle.position.tolist(),
                'battery_level': vehicle.battery_level,
                'current_tasks': len(vehicle.current_tasks),
                'trust_score': vehicle.trust_score
            }
            info['vehicle_states'].append(vehicle_state)
        
        # RSU状态信息
        for rsu in self.rsus:
            rsu_state = {
                'rsu_id': rsu.rsu_id,
                'position': rsu.position.tolist(),
                'connected_vehicles': len(rsu.connected_vehicles),
                'mec_cpu_load': rsu.mec_server.current_cpu_load,
                'task_queue_length': len(rsu.mec_server.task_queue)
            }
            info['rsu_states'].append(rsu_state)
        
        return info
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        total_tasks = self.metrics['completed_tasks'] + self.metrics['failed_tasks']
        
        if total_tasks > 0:
            completion_rate = self.metrics['completed_tasks'] / total_tasks
        else:
            completion_rate = 0.0
        
        if self.metrics['completed_tasks'] > 0:
            avg_latency = self.metrics['total_latency'] / self.metrics['completed_tasks']
            avg_energy = self.metrics['total_energy_consumption'] / self.metrics['completed_tasks']
        else:
            avg_latency = 0.0
            avg_energy = 0.0
        
        return {
            'task_completion_rate': completion_rate,
            'average_latency': avg_latency,
            'average_energy_consumption': avg_energy,
            'total_v2v_collaborations': self.metrics['v2v_collaborations'],
            'average_mec_utilization': np.mean(self.metrics['mec_utilization']) if self.metrics['mec_utilization'] else 0.0
        }

# 示例配置
DEFAULT_CONFIG = {
    'num_vehicles': 10,
    'num_rsus': 4,
    'map_size': 1000.0,
    'time_step': 0.1,
    'task_generation_prob': 0.3,
    'simulation_duration': 100.0
}

if __name__ == "__main__":
    # 测试仿真环境
    sim = CompleteMECV2XSimulation(DEFAULT_CONFIG)
    
    # 重置环境
    observations = sim.reset()
    print(f"初始观测维度: {observations['vehicle_0'].shape}")
    
    # 运行几个步骤
    for step in range(10):
        # 随机动作
        actions = {}
        for i in range(sim.config['num_vehicles']):
            actions[f'vehicle_{i}'] = np.random.uniform(0, 1, 5)
        
        obs, rewards, dones, info = sim.step(actions)
        
        print(f"Step {step}:")
        print(f"  平均奖励: {np.mean(list(rewards.values())):.4f}")
        print(f"  活跃连接数: {info['communication_stats']['active_connections']}")
        print(f"  MEC平均利用率: {info['communication_stats']['average_mec_utilization']:.4f}")
    
    # 显示性能指标
    metrics = sim.get_performance_metrics()
    print(f"\n性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}") 