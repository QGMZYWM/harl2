import torch
import torch.nn as nn
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP


class DeterministicPolicy(nn.Module):
    """Deterministic policy network for HADDPG and HATD3."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape, hidden_sizes[0], activation_func
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]
        act_dim = action_space.shape[0]
        self.net = PlainMLP(
            [feature_dim] + list(hidden_sizes) + [act_dim],
            activation_func,
            final_activation_func,
        )
        
        # 安全地获取action_space.high
        try:
            # 尝试直接转换整个high数组
            high = torch.tensor(action_space.high, dtype=torch.float32).to(device)
        except:
            # 如果失败，尝试逐个元素转换
            high_list = []
            for i in range(act_dim):
                try:
                    high_list.append(action_space.high[i])
                except:
                    high_list.append(action_space.high)
            high = torch.tensor(high_list, dtype=torch.float32).to(device)
            
        self.register_buffer("high", high)
        self.to(device)

    def forward(self, obs):
        """Return output from network scaled to action space limits."""
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        action = self.net(x)
        action = self.high * action
        return action
