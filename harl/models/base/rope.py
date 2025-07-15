import torch
import torch.nn as nn
import math


class RotaryPositionEncoding(nn.Module):
    """
    旋转位置编码（Rotary Position Encoding, RoPE）模块
    
    RoPE通过在特征空间中应用旋转变换来编码相对位置信息，
    具有更好的长序列泛化能力和外推能力。
    
    参考文献: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim, base=10000.0, scale_base=None):
        """
        初始化RoPE模块
        
        Args:
            dim: 特征维度，必须是偶数
            base: 频率基数，控制旋转的速度
            scale_base: 缩放基数，用于RoPE-scaling，默认为None（不使用缩放）
        """
        super().__init__()
        assert dim % 2 == 0, "特征维度必须是偶数"
        
        # 计算不同频率的theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 缓存变量
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
        # RoPE缩放参数
        self.scale_base = scale_base
        
    def _update_cos_sin_cache(self, x, seq_dim=1):
        """
        更新cos和sin缓存
        
        Args:
            x: 输入张量
            seq_dim: 序列维度的索引
        """
        seq_len = x.shape[seq_dim]
        
        # 如果序列长度变化或缓存未初始化，则更新缓存
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            
            # 创建位置索引
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            
            # 如果使用RoPE缩放
            if self.scale_base is not None:
                t = t / self.scale_base
            
            # 计算频率
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
            
            # 计算cos和sin值
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)  # [seq_len, dim]
            
            # 重新排列为正确的形状以便广播
            if seq_dim == 1:
                # 适用于[batch, seq, head, head_dim]
                self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
                self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
            else:
                # 适用于其他情况
                self.cos_cached = emb.cos()
                self.sin_cached = emb.sin()
    
    def forward(self, x, seq_dim=1):
        """
        应用旋转位置编码
        
        Args:
            x: 输入张量，形状为 [..., seq_len, ..., dim]
            seq_dim: 序列维度的索引
            
        Returns:
            应用了RoPE的张量，形状与输入相同
        """
        # 更新缓存
        self._update_cos_sin_cache(x, seq_dim)
        
        # 确保dim是x的最后一个维度
        orig_shape = x.shape
        
        # 重塑为可以应用旋转的形状
        if seq_dim != 1 or len(x.shape) != 4:
            # 处理非标准形状
            # 将序列维度移到第1个位置，特征维度移到最后
            dims = list(range(len(x.shape)))
            dims.remove(seq_dim)
            last_dim = dims[-1]
            dims.append(seq_dim)
            x = x.permute(*dims)
        
        # 将特征维度分成两半，准备旋转
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        
        # 分离实部和虚部（相当于将每对连续的特征视为复数的实部和虚部）
        x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
        
        # 应用旋转变换
        # 对应复数乘法: (a+bi)(cos+sin*i) = (a*cos-b*sin) + (a*sin+b*cos)i
        rotated_x1 = x1 * self.cos_cached - x2 * self.sin_cached
        rotated_x2 = x1 * self.sin_cached + x2 * self.cos_cached
        
        # 重新组合旋转后的特征
        x_rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        x_rotated = x_rotated.reshape(*x.shape)
        
        # 如果需要，恢复原始形状
        if seq_dim != 1 or len(orig_shape) != 4:
            # 恢复原始维度顺序
            inv_dims = list(range(len(x_rotated.shape)))
            seq_idx = inv_dims.pop()
            inv_dims.insert(seq_dim, seq_idx)
            x_rotated = x_rotated.permute(*inv_dims)
        
        return x_rotated


class RotaryEmbedding(nn.Module):
    """
    旋转位置嵌入的简化版本，专为Transformer的多头注意力设计
    
    这个版本针对标准的Transformer架构进行了优化，
    假设输入形状为 [batch_size, seq_len, num_heads, head_dim]
    """
    
    def __init__(self, dim, base=10000.0):
        """
        初始化旋转位置嵌入
        
        Args:
            dim: 头维度 (head_dim)
            base: 频率基数
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, q, k, seq_len=None):
        """
        对查询和键应用旋转位置编码
        
        Args:
            q: 查询张量 [batch_size, seq_len, num_heads, head_dim]
            k: 键张量 [batch_size, seq_len, num_heads, head_dim]
            seq_len: 序列长度，如果为None则从q中推断
            
        Returns:
            q_rot: 旋转后的查询
            k_rot: 旋转后的键
        """
        if seq_len is None:
            seq_len = q.shape[1]  # 假设q的形状是[batch, seq, head, dim]
        
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(q.device)
            
            # [seq_len, 1, 1, dim]
            self.cos_cached = emb.cos().view(seq_len, 1, 1, -1)
            self.sin_cached = emb.sin().view(seq_len, 1, 1, -1)
        
        # 确保cos和sin缓存的维度与q和k匹配
        if self.cos_cached is None or self.sin_cached is None:
            # 如果缓存为空，先更新缓存
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(q.device)
            
            # [seq_len, 1, 1, dim]
            self.cos_cached = emb.cos().view(seq_len, 1, 1, -1)
            self.sin_cached = emb.sin().view(seq_len, 1, 1, -1)
            
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # 将q和k分成实部和虚部
        q_dim = q.shape[-1]
        k_dim = k.shape[-1]
        
        # 确保维度是偶数
        assert q_dim % 2 == 0 and k_dim % 2 == 0, "头维度必须是偶数"
        
        # 重塑为可旋转的形式
        q_reshape = q.reshape(*q.shape[:-1], -1, 2)
        k_reshape = k.reshape(*k.shape[:-1], -1, 2)
        
        q1, q2 = q_reshape[..., 0], q_reshape[..., 1]
        k1, k2 = k_reshape[..., 0], k_reshape[..., 1]
        
        # 应用旋转
        q_rot1 = q1 * cos - q2 * sin
        q_rot2 = q1 * sin + q2 * cos
        k_rot1 = k1 * cos - k2 * sin
        k_rot2 = k1 * sin + k2 * cos
        
        # 重新组合
        q_rot = torch.stack([q_rot1, q_rot2], dim=-1).reshape(*q.shape)
        k_rot = torch.stack([k_rot1, k_rot2], dim=-1).reshape(*k.shape)
        
        return q_rot, k_rot 