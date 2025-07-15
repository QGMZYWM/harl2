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
            output: 编码器输出 [batch_size, seq_len, d_model] (因为batch_first=True)
            attention_weights: 注意力权重 [num_layers, batch_size, num_heads, seq_len, seq_len]
        """
        output = src
        attention_weights_list = []
        
        # 获取Transformer编码器的每一层
        for i, layer in enumerate(self.transformer_encoder.layers):
            # 保存原始的自注意力模块
            original_self_attn = layer.self_attn
            
            # 临时替换为我们的自定义注意力模块
            layer.self_attn = self.custom_attention_layers[i]
            
            # 执行单层Transformer编码
            # 注意：这里使用的是标准的层级处理，但使用我们的自定义注意力
            src2 = layer.self_attn(output, output, output,
                                  key_padding_mask=src_key_padding_mask)
            
            # 提取注意力权重
            attn_weights = layer.self_attn.last_attn_weights
            attention_weights_list.append(attn_weights)
            
            # 执行剩余的前向传播步骤
            output = output + layer.dropout1(src2)
            output = layer.norm1(output)
            
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            output = output + layer.dropout2(src2)
            output = layer.norm2(output)
            
            # 恢复原始的自注意力模块
            layer.self_attn = original_self_attn
        
        # 堆叠所有层的注意力权重
        if attention_weights_list:
            attention_weights = torch.stack(attention_weights_list)  # [num_layers, batch, num_heads, seq_len, seq_len]
        else:
            # 如果没有捕获到注意力权重，创建一个空张量
            batch_size, seq_len, _ = src.size()
            attention_weights = torch.zeros(
                self.num_layers, batch_size, self.nhead, seq_len, seq_len,
                device=self.device
            )
        
        return output, attention_weights
    
    def get_context_embedding_dim(self):
        """返回上下文嵌入的维度"""
        return self.d_model


class CustomMultiheadAttention(nn.Module):
    """
    自定义多头注意力模块，可以捕获注意力权重
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, use_rope=True, rope_base=10000.0, rope_scale_base=None):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 是否使用RoPE
        self.use_rope = use_rope
        if use_rope:
            # 导入RoPE模块
            from harl.models.base.rope import RotaryEmbedding
            self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        
        self.last_attn_weights = None
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        执行多头自注意力计算，并保存注意力权重
        
        Args:
            query: 查询张量 [batch_size, seq_len, embed_dim] (因为batch_first=True)
            key: 键张量 [batch_size, seq_len, embed_dim]
            value: 值张量 [batch_size, seq_len, embed_dim]
            key_padding_mask: 键填充掩码 [batch_size, seq_len]
            
        Returns:
            attn_output: 注意力输出 [batch_size, seq_len, embed_dim]
        """
        bsz, seq_len, embed_dim = query.size()
        
        # 线性投影
        q = self.q_proj(query).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        k = self.k_proj(key).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # [bsz, num_heads, seq_len, head_dim]
        v = self.v_proj(value).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        
        # 如果使用RoPE，应用旋转位置编码到q和k
        if self.use_rope:
            q, k = self.rope(q, k, seq_len=seq_len)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用填充掩码
        if key_padding_mask is not None:
            # 将掩码扩展到适当的维度
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [bsz, 1, 1, seq_len]
            scores = scores.masked_fill(expanded_mask, float('-inf'))
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [bsz, num_heads, seq_len, seq_len]
        
        # 保存注意力权重供后续分析
        self.last_attn_weights = attn_weights.detach()
        
        # 应用dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)  # [bsz, num_heads, seq_len, head_dim]
        
        # 重塑并连接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, embed_dim)  # [bsz, seq_len, embed_dim]
        
        # 最终线性投影
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class HistoryBuffer:
    """
    用于存储和管理智能体历史观测-动作序列的缓冲区
    """
    
    def __init__(self, max_length, obs_dim, action_dim, device=torch.device("cpu")):
        self.max_length = max_length
        self.obs_dim = obs_dim  
        self.action_dim = action_dim
        self.device = device
        
        # 初始化缓冲区
        self.obs_buffer = torch.zeros(max_length, obs_dim, device=device)
        self.action_buffer = torch.zeros(max_length, action_dim, device=device)
        self.current_length = 0
        self.current_idx = 0
    
    def add(self, obs, action):
        """添加新的观测-动作对"""
        self.obs_buffer[self.current_idx] = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self.action_buffer[self.current_idx] = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        self.current_idx = (self.current_idx + 1) % self.max_length
        self.current_length = min(self.current_length + 1, self.max_length)
    
    def get_sequence(self):
        """获取当前存储的序列"""
        if self.current_length == 0:
            return None, None, 0
        
        if self.current_length < self.max_length:
            # 缓冲区还没满
            obs_seq = self.obs_buffer[:self.current_length]
            action_seq = self.action_buffer[:self.current_length] 
        else:
            # 缓冲区已满，需要按正确顺序重排
            obs_seq = torch.cat([
                self.obs_buffer[self.current_idx:],
                self.obs_buffer[:self.current_idx]
            ], dim=0)
            action_seq = torch.cat([
                self.action_buffer[self.current_idx:],
                self.action_buffer[:self.current_idx]
            ], dim=0)
        
        return obs_seq.unsqueeze(0), action_seq.unsqueeze(0), self.current_length
    
    def reset(self):
        """重置缓冲区"""
        self.current_length = 0
        self.current_idx = 0
        self.obs_buffer.zero_()
        self.action_buffer.zero_()
