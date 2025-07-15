"""PopArt: Preserving Outputs Precisely, while Adaptively Rescaling Targets."""

import torch
import torch.nn as nn
import numpy as np


class PopArt(nn.Module):
    """PopArt normalization layer.
    
    PopArt (Preserving Outputs Precisely, while Adaptively Rescaling Targets) 
    is a technique for normalizing targets in reinforcement learning.
    
    Reference: https://arxiv.org/abs/1602.07714
    """
    
    def __init__(self, input_shape, beta=0.99, epsilon=1e-5):
        """Initialize PopArt layer.
        
        Args:
            input_shape: Shape of the input tensor
            beta: Exponential moving average decay
            epsilon: Small value for numerical stability
        """
        super(PopArt, self).__init__()
        
        self.beta = beta
        self.epsilon = epsilon
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(input_shape))
        self.register_buffer('running_var', torch.ones(input_shape))
        self.register_buffer('count', torch.zeros(1))
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(input_shape))
        self.bias = nn.Parameter(torch.zeros(input_shape))
        
    def forward(self, x):
        """Forward pass."""
        if self.training:
            # Update running statistics
            self.update_stats(x)
            
        # Normalize
        normalized = (x - self.running_mean) / (torch.sqrt(self.running_var) + self.epsilon)
        
        # Apply learnable transformation
        return normalized * self.weight + self.bias
    
    def update_stats(self, x):
        """Update running statistics."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        # Update count
        self.count += batch_count
        
        # Update running mean and variance
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_count / self.count
        
        # Update running variance
        delta2 = batch_var - self.running_var
        self.running_var += delta2 * batch_count / self.count
        
    def normalize(self, x):
        """Normalize targets."""
        return (x - self.running_mean) / (torch.sqrt(self.running_var) + self.epsilon)
        
    def denormalize(self, x):
        """Denormalize values."""
        return x * (torch.sqrt(self.running_var) + self.epsilon) + self.running_mean
    
    def update_parameters(self, new_mean, new_std):
        """Update parameters when target distribution changes."""
        old_std = torch.sqrt(self.running_var) + self.epsilon
        old_mean = self.running_mean
        
        # Update weight and bias to preserve outputs
        self.weight.data = self.weight.data * old_std / new_std
        self.bias.data = (self.bias.data * old_std + old_mean - new_mean) / new_std
        
        # Update running statistics
        self.running_mean.data = new_mean
        self.running_var.data = new_std ** 2
        
    def reset_parameters(self):
        """Reset parameters to initial values."""
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.running_mean.data.fill_(0.0)
        self.running_var.data.fill_(1.0)
        self.count.data.fill_(0.0) 