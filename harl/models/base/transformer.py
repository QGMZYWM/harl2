import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块，为Transformer提供序列位置信息"""
    
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """
    用于V2X动态上下文感知状态表征的Transformer编码器
    
    这个模块处理智能体的历史观测-动作序列，生成丰富的上下文感知状态嵌入
    """
    
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        """
        Args:
            args: 配置参数字典
            obs_dim: 观测维度
            action_dim: 动作维度  
            device: 计算设备
        """
        super(TransformerEncoder, self).__init__()
        
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 从args中获取参数，设置默认值
        self.d_model = args.get("transformer_d_model", 256)
        self.nhead = args.get("transformer_nhead", 8)
        self.num_layers = args.get("transformer_num_layers", 4)
        self.dim_feedforward = args.get("transformer_dim_feedforward", 512)
        self.dropout = args.get("transformer_dropout", 0.1)
        self.max_seq_length = args.get("max_seq_length", 50)
        
        # 使用RoPE还是标准位置编码
        self.use_rope = args.get("use_rope", True)  # 默认使用RoPE
        self.rope_base = args.get("rope_base", 10000.0)  # RoPE频率基数
        self.rope_scale_base = args.get("rope_scale_base", None)  # RoPE缩放基数
        
        # 输入投影层：将观测-动作拼接向量投影到d_model维度
        input_dim = obs_dim + action_dim
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # 位置编码 - 根据配置选择使用RoPE或标准位置编码
        if not self.use_rope:
            self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu',
            batch_first=True  # 添加batch_first=True参数
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # 输出投影层：将最终隐藏状态投影到期望的输出维度
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # 用于存储注意力权重的属性
        self.last_attention_weights = None
        
        # 创建自定义的多头注意力模块来替换原始模块
        self.custom_attention_layers = nn.ModuleList([
            CustomMultiheadAttention(
                self.d_model, 
                self.nhead, 
                dropout=self.dropout, 
                use_rope=self.use_rope,
                rope_base=self.rope_base,
                rope_scale_base=self.rope_scale_base
            )
            for _ in range(self.num_layers)
        ])
        
        self.to(device)
    
    def forward(self, obs_seq, action_seq, seq_lengths=None):
        """
        前向传播
        
        Args:
            obs_seq: 观测序列 [batch_size, seq_len, obs_dim]
            action_seq: 动作序列 [batch_size, seq_len, action_dim] 
            seq_lengths: 每个序列的实际长度 [batch_size] (可选)
            
        Returns:
            context_embedding: 上下文感知状态嵌入 [batch_size, d_model]
            sequence_embeddings: 序列中每个时间步的嵌入 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = obs_seq.shape
        
        # 拼接观测和动作
        obs_action = torch.cat([obs_seq, action_seq], dim=-1)  # [batch, seq_len, obs_dim + action_dim]
        
        # 输入投影
        embedded = self.input_projection(obs_action)  # [batch, seq_len, d_model]
        
        # 添加位置编码 - 仅当不使用RoPE时
        if not self.use_rope:
            # 由于batch_first=True，我们需要先转换为[seq_len, batch, d_model]，添加位置编码后再转回来
            embedded = embedded.transpose(0, 1)  # [seq_len, batch, d_model]
            embedded = self.pos_encoder(embedded)  # 添加位置编码
            embedded = embedded.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # 创建padding mask（如果提供了序列长度）
        src_key_padding_mask = None
        if seq_lengths is not None:
            if isinstance(seq_lengths, int):
                # 如果seq_lengths是单个整数，创建全部长度相同的mask
                if seq_lengths < seq_len:
                    src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
                    src_key_padding_mask[:, seq_lengths:] = True
            else:
                # 如果是张量或列表，为每个序列创建独立的mask
                src_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
                for i, length in enumerate(seq_lengths):
                    if length < seq_len:
                        src_key_padding_mask[i, length:] = True
        
        # 使用自定义的前向传播来捕获注意力权重
        transformer_output, attention_weights = self._forward_with_attention(
            embedded, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 保存注意力权重供定性分析使用
        self.last_attention_weights = attention_weights
        
        # 不再需要转换格式，因为batch_first=True
        # transformer_output = transformer_output.transpose(0, 1)
        
        # 获取序列表示（使用最后一个时间步的输出或平均池化）
        if isinstance(seq_lengths, int):
            # 使用固定的最后一个有效时间步
            if seq_lengths < seq_len:
                context_embedding = transformer_output[:, seq_lengths - 1]
            else:
                context_embedding = transformer_output[:, -1]
        elif seq_lengths is not None:
            # 使用每个序列的最后一个有效时间步
            indices = torch.tensor([min(l-1, seq_len-1) for l in seq_lengths], device=self.device)
            batch_indices = torch.arange(batch_size, device=self.device)
            context_embedding = transformer_output[batch_indices, indices]
        else:
            # 使用最后一个时间步
            context_embedding = transformer_output[:, -1, :]  # [batch, d_model]
        
        # 应用输出投影和归一化
        context_embedding = self.output_projection(context_embedding)
        context_embedding = self.layer_norm(context_embedding)
        
        # 同时返回每个时间步的嵌入（用于对比学习）
        sequence_embeddings = self.output_projection(transformer_output)
        sequence_embeddings = self.layer_norm(sequence_embeddings)
        
        return context_embedding, sequence_embeddings
    
    def _forward_with_attention(self, src, src_key_padding_mask=None):
        """
        自定义前向传播，捕获注意力权重
        
        Args:
            src: 输入序列 [batch_size, seq_len, d_model] (因为batch_first=True)
            src_key_padding_mask: 填充掩码 [batch_size, seq_len]
            
        Returns:
            output: Transformer编码器输出 [batch_size, seq_len, d_model]
            attention_weights: 注意力权重 [num_layers, batch_size, num_heads, seq_len, seq_len]
        """
        # 形状检查
        if src.dim() != 3:
            raise ValueError(f"输入序列维度错误: 期望3维, 实际{src.dim()}维")
        
        batch_size, seq_len, d_model = src.shape
        if d_model != self.d_model:
            raise ValueError(f"输入特征维度不匹配: 期望{self.d_model}, 实际{d_model}")
        
        # 初始化输出和注意力权重
        output = src
        attention_weights = []
        
        # 逐层处理
        for i, attention_layer in enumerate(self.custom_attention_layers):
            # 应用自注意力
            attn_output, attn_weights = attention_layer(
                output, output, output,
                key_padding_mask=src_key_padding_mask
            )
            
            # 残差连接和归一化
            output = output + attn_output
            output = F.layer_norm(output, [self.d_model])
            
            # 前馈网络
            ff_output = F.linear(output, 
                                self.transformer_encoder.layers[i].linear1.weight,
                                self.transformer_encoder.layers[i].linear1.bias)
            ff_output = F.relu(ff_output)
            ff_output = F.dropout(ff_output, p=self.dropout, training=self.training)
            ff_output = F.linear(ff_output, 
                               self.transformer_encoder.layers[i].linear2.weight,
                               self.transformer_encoder.layers[i].linear2.bias)
            
            # 残差连接和归一化
            output = output + ff_output
            output = F.layer_norm(output, [self.d_model])
            
            # 收集注意力权重
            attention_weights.append(attn_weights)
        
        # 将注意力权重堆叠为单个张量 [num_layers, batch_size, num_heads, seq_len, seq_len]
        if attention_weights:
            try:
                attention_weights = torch.stack(attention_weights)
            except:
                # 如果形状不一致，使用空张量
                attention_weights = torch.zeros(
                    self.num_layers, batch_size, self.nhead, seq_len, seq_len,
                    device=src.device
                )
        else:
            attention_weights = torch.zeros(
                self.num_layers, batch_size, self.nhead, seq_len, seq_len,
                device=src.device
            )
        
        return output, attention_weights
    
    def get_context_embedding_dim(self):
        """返回上下文嵌入的维度"""
        return self.d_model


class CustomMultiheadAttention(nn.Module):
    """
    自定义多头注意力模块，支持RoPE位置编码
    
    与标准nn.MultiheadAttention不同，这个模块会保存注意力权重以供可视化
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rope=True, rope_base=10000.0, rope_scale_base=None):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout概率
            use_rope: 是否使用RoPE位置编码
            rope_base: RoPE频率基数
            rope_scale_base: RoPE缩放基数
        """
        super(CustomMultiheadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.rope_scale_base = rope_scale_base
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"嵌入维度 {embed_dim} 必须能被头数 {num_heads} 整除")
        
        # 投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 用于存储最后一次的注意力权重
        self.last_attn_weights = None
    
    def forward(self, query, key, value, key_padding_mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, embed_dim]
            key: 键张量 [batch_size, seq_len, embed_dim]
            value: 值张量 [batch_size, seq_len, embed_dim]
            key_padding_mask: 键填充掩码 [batch_size, seq_len]
            
        Returns:
            attn_output: 注意力输出 [batch_size, seq_len, embed_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
        """
        # 形状检查
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(f"查询、键和值必须是3维张量")
        
        batch_size, tgt_len, _ = query.size()
        _, src_len, _ = key.size()
        
        # 投影查询、键、值
        q = self.q_proj(query)  # [batch_size, tgt_len, embed_dim]
        k = self.k_proj(key)    # [batch_size, src_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, src_len, embed_dim]
        
        # 重塑为多头形状
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 应用RoPE位置编码（如果启用）
        if self.use_rope:
            q, k = self._apply_rotary_position_embeddings(q, k, tgt_len, src_len)
        
        # 计算注意力分数
        # [batch_size, num_heads, tgt_len, src_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # 转换掩码形状以匹配注意力权重
            # [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask, float('-inf')
            )
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 存储注意力权重供可视化使用
        self.last_attn_weights = attn_weights.detach()
        
        # 应用注意力权重到值
        # [batch_size, num_heads, tgt_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置回原始形状
        # [batch_size, tgt_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 合并多头
        # [batch_size, tgt_len, embed_dim]
        attn_output = attn_output.view(batch_size, tgt_len, self.embed_dim)
        
        # 最终投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
    
    def _apply_rotary_position_embeddings(self, q, k, q_len, k_len):
        """
        应用旋转位置编码（RoPE）
        
        Args:
            q: 查询张量 [batch_size, num_heads, q_len, head_dim]
            k: 键张量 [batch_size, num_heads, k_len, head_dim]
            q_len: 查询序列长度
            k_len: 键序列长度
            
        Returns:
            q_rope: 应用RoPE后的查询 [batch_size, num_heads, q_len, head_dim]
            k_rope: 应用RoPE后的键 [batch_size, num_heads, k_len, head_dim]
        """
        device = q.device
        
        # 确保head_dim是偶数
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE要求head_dim为偶数，但得到了{self.head_dim}")
        
        # 生成位置索引
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)  # [q_len, 1]
        k_pos = torch.arange(k_len, device=device).unsqueeze(1)  # [k_len, 1]
        
        # 生成频率因子
        freq_seq = torch.arange(0, self.head_dim, 2, device=device).float()  # [head_dim/2]
        inv_freq = 1.0 / (self.rope_base ** (freq_seq / self.head_dim))  # [head_dim/2]
        
        # 计算位置编码
        q_freqs = torch.matmul(q_pos, inv_freq.unsqueeze(0))  # [q_len, head_dim/2]
        k_freqs = torch.matmul(k_pos, inv_freq.unsqueeze(0))  # [k_len, head_dim/2]
        
        # 应用RoPE
        q_rope = self._apply_rotary_embedding(q, q_freqs)
        k_rope = self._apply_rotary_embedding(k, k_freqs)
        
        return q_rope, k_rope
    
    def _apply_rotary_embedding(self, x, freqs):
        """
        应用旋转位置编码到张量
        
        Args:
            x: 输入张量 [batch_size, num_heads, seq_len, head_dim]
            freqs: 频率 [seq_len, head_dim/2]
            
        Returns:
            x_rope: 应用RoPE后的张量 [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # 重塑为 [batch_size * num_heads, seq_len, head_dim]
        x_flat = x.reshape(batch_size * num_heads, seq_len, head_dim)
        
        # 分离实部和虚部
        x_real = x_flat[..., 0::2]  # [batch*num_heads, seq_len, head_dim/2]
        x_imag = x_flat[..., 1::2]  # [batch*num_heads, seq_len, head_dim/2]
        
        # 计算旋转
        cos = torch.cos(freqs).unsqueeze(0)  # [1, seq_len, head_dim/2]
        sin = torch.sin(freqs).unsqueeze(0)  # [1, seq_len, head_dim/2]
        
        # 应用复数乘法
        x_real_rot = x_real * cos - x_imag * sin
        x_imag_rot = x_real * sin + x_imag * cos
        
        # 交错合并实部和虚部
        x_rope = torch.zeros_like(x_flat)
        x_rope[..., 0::2] = x_real_rot
        x_rope[..., 1::2] = x_imag_rot
        
        # 恢复原始形状
        x_rope = x_rope.reshape(batch_size, num_heads, seq_len, head_dim)
        
        return x_rope


class HistoryBuffer:
    """
    历史观测-动作序列缓冲区
    
    用于存储和管理智能体的历史观测和动作，供Transformer处理
    支持批量操作，可以同时处理多个样本
    """
    
    def __init__(self, max_length, obs_dim, action_dim, device=torch.device("cpu")):
        """
        Args:
            max_length: 最大序列长度
            obs_dim: 观测维度
            action_dim: 动作维度
            device: 计算设备
        """
        self.max_length = max_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # 初始化空缓冲区
        self.obs_buffer = []
        self.action_buffer = []
    
    def add(self, obs, action):
        """
        添加观测和动作到历史缓冲区
        
        Args:
            obs: 观测 [batch_size, obs_dim]
            action: 动作 [batch_size, action_dim]
        """
        # 确保输入是张量
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            
        # 确保形状正确
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [obs_dim] -> [1, obs_dim]
        if action.dim() == 1:
            action = action.unsqueeze(0)  # [action_dim] -> [1, action_dim]
            
        # 添加到缓冲区
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        
        # 如果超过最大长度，移除最旧的
        if len(self.obs_buffer) > self.max_length:
            self.obs_buffer.pop(0)
            self.action_buffer.pop(0)
    
    def get_sequence(self):
        """
        获取历史序列
        
        Returns:
            obs_seq: 观测序列 [batch_size, seq_len, obs_dim]
            action_seq: 动作序列 [batch_size, seq_len, action_dim]
            seq_length: 序列长度
        """
        if not self.obs_buffer:
            # 如果缓冲区为空，返回空序列
            batch_size = 1
            return (
                torch.zeros(batch_size, 1, self.obs_dim, device=self.device),
                torch.zeros(batch_size, 1, self.action_dim, device=self.device),
                1
            )
        
        # 获取批次大小
        batch_size = self.obs_buffer[-1].shape[0]
        seq_length = len(self.obs_buffer)
        
        # 初始化序列张量
        obs_seq = torch.zeros(batch_size, seq_length, self.obs_dim, device=self.device)
        action_seq = torch.zeros(batch_size, seq_length, self.action_dim, device=self.device)
        
        # 填充序列
        for i, (obs, action) in enumerate(zip(self.obs_buffer, self.action_buffer)):
            # 确保每个批次的大小一致
            if obs.shape[0] != batch_size:
                # 如果批次大小不一致，复制最后一个批次的数据
                obs = self.obs_buffer[-1].clone()
                action = self.action_buffer[-1].clone()
                
            obs_seq[:, i, :] = obs
            action_seq[:, i, :] = action
        
        return obs_seq, action_seq, seq_length
    
    def reset(self):
        """重置缓冲区"""
        self.obs_buffer = []
        self.action_buffer = []
