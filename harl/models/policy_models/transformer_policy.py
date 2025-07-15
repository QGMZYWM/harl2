import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.transformer import TransformerEncoder, HistoryBuffer
from harl.utils.contrastive_learning import EnhancedContrastiveLoss, V2XStatesSimilarity


class TransformerEnhancedPolicy(nn.Module):
    """
    集成Transformer编码器的增强策略网络
    
    实现HASAC-Flow第一个创新点：动态上下文感知状态表征
    通过Transformer编码器处理历史观测序列，结合对比学习优化状态表征
    """
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """
        Args:
            args: 配置参数
            obs_space: 观测空间
            action_space: 动作空间  
            device: 计算设备
        """
        super(TransformerEnhancedPolicy, self).__init__()
        
        self.device = device
        self.args = args
        
        # 获取观测和动作维度
        self.obs_dim = obs_space.shape[0]
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
            self.discrete_action = True
        else:
            self.action_dim = action_space.shape[0] 
            self.discrete_action = False
        
        # Transformer相关参数
        self.max_seq_length = args.get("max_seq_length", 50)
        self.use_transformer = args.get("use_transformer", True)
        self.use_contrastive_learning = args.get("use_contrastive_learning", True)
        
        # 创建Transformer编码器
        if self.use_transformer:
            self.transformer_encoder = TransformerEncoder(
                args, self.obs_dim, self.action_dim, device
            )
            self.context_dim = self.transformer_encoder.get_context_embedding_dim()
        else:
            self.context_dim = self.obs_dim
            
        # 历史缓冲区（每个智能体一个）
        self.history_buffers = {}
        
        # 对比学习损失
        if self.use_contrastive_learning:
            self.contrastive_loss_fn = EnhancedContrastiveLoss(
                temperature=args.get("contrastive_temperature", 0.1),
                similarity_threshold=args.get("similarity_threshold", 0.8),
                temporal_weight=args.get("temporal_weight", 0.1)
            )
        
        # 策略网络（基于上下文嵌入）
        self.policy_network = self._build_policy_network(args)
        
        # 上一时刻的嵌入（用于时间对比学习）
        self.previous_embeddings = None
        
        self.to(device)
    
    def _build_policy_network(self, args):
        """构建策略网络"""
        hidden_size = args.get("hidden_size", 256)
        activation = args.get("activation", "relu")
        
        if activation == "relu":
            act_func = nn.ReLU
        elif activation == "tanh":
            act_func = nn.Tanh
        else:
            act_func = nn.ReLU
        
        if self.discrete_action:
            # 离散动作空间
            layers = [
                nn.Linear(self.context_dim, hidden_size),
                act_func(),
                nn.Linear(hidden_size, hidden_size),
                act_func(),
                nn.Linear(hidden_size, self.action_dim)
            ]
        else:
            # 连续动作空间（输出均值和方差）
            layers = [
                nn.Linear(self.context_dim, hidden_size),
                act_func(),
                nn.Linear(hidden_size, hidden_size),
                act_func(),
                nn.Linear(hidden_size, self.action_dim * 2)  # 均值和log_std
            ]
        
        return nn.Sequential(*layers)
    
    def forward(self, obs, actions=None, agent_id=None, available_actions=None, deterministic=False, return_one_hot=False):
        """
        前向传播
        
        Args:
            obs: 当前观测 [batch_size, obs_dim] 或 [obs_dim]
            actions: 历史动作（用于构建序列）
            agent_id: 智能体ID
            available_actions: 可用动作mask
            deterministic: 是否确定性输出
            return_one_hot: 对于离散动作，是否返回one-hot编码（用于critic训练）
            
        Returns:
            actions: 采样的动作
            action_log_probs: 动作的对数概率
            dist_entropy: 分布熵
            context_embedding: 上下文嵌入（用于对比学习）
        """
        # 处理输入维度
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single_agent = True
        else:
            single_agent = False
        
        batch_size = obs.size(0)
        
        # 获取上下文嵌入
        context_embedding, contrastive_info = self._get_context_embedding(obs, actions, agent_id)
        
        # 通过策略网络生成动作分布
        if self.discrete_action:
            action_logits = self.policy_network(context_embedding)
            
            # 应用可用动作mask
            if available_actions is not None:
                # 确保available_actions在相同设备上
                if not isinstance(available_actions, torch.Tensor):
                    available_actions = torch.tensor(available_actions, device=self.device)
                else:
                    available_actions = available_actions.to(self.device)
                action_logits = action_logits - 1e8 * (1 - available_actions)
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
        else:
            policy_output = self.policy_network(context_embedding)
            action_mean = policy_output[:, :self.action_dim]
            action_log_std = policy_output[:, self.action_dim:]
            action_log_std = torch.clamp(action_log_std, -20, 2)
            action_std = torch.exp(action_log_std)
            
            action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # 采样动作
        if self.discrete_action:
            # 离散动作：先获取动作索引
            if deterministic:
                action_indices = torch.argmax(action_logits, dim=-1)
            else:
                action_indices = action_dist.sample()
            
            # 使用动作索引计算对数概率
            action_log_probs = action_dist.log_prob(action_indices)
            
            # 根据return_one_hot参数决定返回格式
            if return_one_hot:
                # 返回 one-hot 编码（用于 critic 训练）
                actions = torch.zeros(action_indices.size(0), self.action_dim, device=action_indices.device)
                actions.scatter_(1, action_indices.unsqueeze(1), 1)
            else:
                # 返回动作索引（用于环境交互），确保形状为 [batch_size, 1]
                actions = action_indices.unsqueeze(1)
        else:
            # 连续动作
            if deterministic:
                actions = action_dist.mean
            else:
                actions = action_dist.sample()
            
            # 确保形状正确
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            
            # 计算对数概率
            action_log_probs = action_dist.log_prob(actions)
            action_log_probs = action_log_probs.sum(dim=-1)
        
        # 计算熵
        dist_entropy = action_dist.entropy()
        if not self.discrete_action:
            dist_entropy = dist_entropy.sum(dim=-1)
        
        # 如果是单个智能体，去掉batch维度
        if single_agent:
            # 对于 actions，根据格式保持正确维度
            if self.discrete_action:
                if return_one_hot:
                    actions = actions.squeeze(0)  # [1, action_dim] -> [action_dim] 对于 one-hot
                else:
                    actions = actions.squeeze(0)  # [1, 1] -> [1] 对于动作索引
            else:
                actions = actions.squeeze(0)  # [1, action_dim] -> [action_dim] 对于连续动作
            action_log_probs = action_log_probs.squeeze(0)
            dist_entropy = dist_entropy.squeeze(0)
            context_embedding = context_embedding.squeeze(0)
        
        return actions, action_log_probs, dist_entropy, context_embedding, contrastive_info
    
    def _get_context_embedding(self, obs, actions, agent_id):
        """
        获取上下文嵌入
        
        Args:
            obs: 当前观测 [batch_size, obs_dim] 或 [obs_dim]
            actions: 历史动作（可选）
            agent_id: 智能体ID
            
        Returns:
            context_embedding: 上下文嵌入 [batch_size, context_dim]
            contrastive_info: 用于对比学习的信息
        """
        # 形状检查和调整
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [obs_dim] -> [1, obs_dim]
            
        # 检查obs的形状是否为[batch_size, obs_dim]
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.obs_dim}, 实际{obs.shape[-1]}")
            
        batch_size = obs.size(0)
        
        # 如果不使用Transformer，直接返回观测作为上下文
        if not self.use_transformer:
            return obs, None
        
        # 确保agent_id存在
        if agent_id is None:
            agent_id = 0
            
        # 获取或创建历史缓冲区
        if agent_id not in self.history_buffers:
            self.history_buffers[agent_id] = HistoryBuffer(
                self.max_seq_length, self.obs_dim, self.action_dim, self.device
            )
        
        # 将当前观测和动作添加到历史中
        # 如果actions为None，使用零向量
        if actions is None:
            if self.discrete_action:
                dummy_action = torch.zeros(batch_size, self.action_dim, device=self.device)
            else:
                dummy_action = torch.zeros(batch_size, self.action_dim, device=self.device)
            self.history_buffers[agent_id].add(obs, dummy_action)
        else:
            # 确保actions的形状正确
            if actions.dim() == 1:
                if self.discrete_action and actions.size(0) == 1:
                    # 离散动作索引转为one-hot
                    action_one_hot = torch.zeros(1, self.action_dim, device=self.device)
                    action_one_hot[0, actions.item()] = 1.0
                    actions = action_one_hot
                else:
                    actions = actions.unsqueeze(0)  # [action_dim] -> [1, action_dim]
                    
            # 检查actions的形状
            if actions.shape[-1] != self.action_dim and not (self.discrete_action and actions.shape[-1] == 1):
                raise ValueError(f"动作维度不匹配: 期望{self.action_dim}, 实际{actions.shape[-1]}")
                
            self.history_buffers[agent_id].add(obs, actions)
        
        # 获取历史序列
        obs_seq, action_seq, seq_length = self.history_buffers[agent_id].get_sequence()
        
        # 检查序列形状
        if obs_seq.shape[0] != batch_size or obs_seq.shape[2] != self.obs_dim:
            raise ValueError(f"观测序列形状错误: {obs_seq.shape}, 期望[{batch_size}, seq_len, {self.obs_dim}]")
        if action_seq.shape[0] != batch_size or action_seq.shape[2] != self.action_dim:
            raise ValueError(f"动作序列形状错误: {action_seq.shape}, 期望[{batch_size}, seq_len, {self.action_dim}]")
        
        # 通过Transformer编码器处理序列
        context_embedding, sequence_embeddings = self.transformer_encoder(
            obs_seq, action_seq, seq_length
        )
        
        # 准备对比学习信息
        contrastive_info = None
        if self.use_contrastive_learning:
            # 保存当前嵌入用于下一时刻的对比学习
            current_embeddings = sequence_embeddings.detach()  # 分离梯度
            
            # 构建对比学习信息
            contrastive_info = {
                'current_embeddings': current_embeddings,
                'previous_embeddings': self.previous_embeddings,
                'seq_length': seq_length
            }
            
            # 更新上一时刻的嵌入
            self.previous_embeddings = current_embeddings
            
            # 保存对比学习信息供评估使用
            self.previous_contrastive_info = contrastive_info
        
        return context_embedding, contrastive_info
    
    def compute_contrastive_loss(self, contrastive_info):
        """计算对比学习损失"""
        if not self.use_contrastive_learning or contrastive_info is None:
            return torch.tensor(0.0, device=self.device)
        
        # FIX: 使用正确的键名来解析contrastive_info字典
        if 'context_embedding' not in contrastive_info:
            return torch.tensor(0.0, device=self.device)
            
        context_embedding = contrastive_info['context_embedding']
        states_info = contrastive_info.get('states_info')
        previous_embedding = contrastive_info.get('previous_embedding')
        
        total_loss, spatial_loss, temporal_loss = self.contrastive_loss_fn(
            context_embedding, states_info, previous_embedding
        )
        
        return total_loss
    
    def evaluate_actions(self, obs, actions, agent_id=None, available_actions=None):
        """
        评估动作的概率和熵（用于训练）
        
        Args:
            obs: 观测
            actions: 要评估的动作
            agent_id: 智能体ID
            available_actions: 可用动作mask
            
        Returns:
            action_log_probs: 动作对数概率
            dist_entropy: 分布熵
            context_embedding: 上下文嵌入
            contrastive_info: 对比学习信息
        """
        # 获取上下文嵌入
        context_embedding, contrastive_info = self._get_context_embedding(obs, None, agent_id)
        
        # 通过策略网络生成动作分布
        if self.discrete_action:
            action_logits = self.policy_network(context_embedding)
            
            if available_actions is not None:
                # 确保available_actions在相同设备上
                if not isinstance(available_actions, torch.Tensor):
                    available_actions = torch.tensor(available_actions, device=self.device)
                else:
                    available_actions = available_actions.to(self.device)
                action_logits = action_logits - 1e8 * (1 - available_actions)
            
            action_dist = torch.distributions.Categorical(logits=action_logits)
        else:
            policy_output = self.policy_network(context_embedding)
            action_mean = policy_output[:, :self.action_dim]
            action_log_std = policy_output[:, self.action_dim:]
            action_log_std = torch.clamp(action_log_std, -20, 2)
            action_std = torch.exp(action_log_std)
            
            action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # 计算给定动作的对数概率和熵
        action_log_probs = action_dist.log_prob(actions)
        if not self.discrete_action:
            action_log_probs = action_log_probs.sum(dim=-1)
        
        dist_entropy = action_dist.entropy()
        if not self.discrete_action:
            dist_entropy = dist_entropy.sum(dim=-1)
        
        return action_log_probs, dist_entropy, context_embedding, contrastive_info
    
    def reset_history(self, agent_id=None):
        """重置历史缓冲区"""
        if agent_id is not None and agent_id in self.history_buffers:
            self.history_buffers[agent_id].reset()
        elif agent_id is None:
            # 重置所有缓冲区
            for buffer in self.history_buffers.values():
                buffer.reset()
        
        # 重置上一时刻的嵌入
        self.previous_embeddings = None


class TransformerActorCritic(nn.Module):
    """
    集成Transformer的Actor-Critic网络
    """
    
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(TransformerActorCritic, self).__init__()
        
        # Actor网络（策略网络）
        self.actor = TransformerEnhancedPolicy(args, obs_space, action_space, device)
        
        # Critic网络（价值网络）
        obs_dim = obs_space.shape[0]
        hidden_size = args.get("hidden_size", 256)
        
        # 使用与actor相同的上下文维度
        if args.get("use_transformer", True):
            context_dim = self.actor.context_dim
        else:
            context_dim = obs_dim
            
        self.critic = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.to(device)
    
    def get_value(self, obs, agent_id=None):
        """获取状态价值"""
        context_embedding, _ = self.actor._get_context_embedding(obs, None, agent_id)
        return self.critic(context_embedding)
    
    def get_action_and_value(self, obs, agent_id=None, available_actions=None, deterministic=False):
        """同时获取动作和价值"""
        action, action_log_prob, entropy, context_embedding, contrastive_info = self.actor(
            obs, agent_id=agent_id, available_actions=available_actions, deterministic=deterministic
        )
        
        value = self.critic(context_embedding)
        
        return action, action_log_prob, entropy, value, contrastive_info
