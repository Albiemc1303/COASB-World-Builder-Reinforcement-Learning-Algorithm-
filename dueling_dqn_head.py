# caosb_world_model/modules/dueling_dqn_head.py

import torch
import torch.nn as nn
from noisy_layers import NoisyLinear

class DuelingDQNHead(nn.Module):
    """
    A Dueling DQN head which takes features as input and outputs Q-values.
    
    It consists of a value stream and an advantage stream. The NoisyLinear
    layers are used for efficient exploration.
    """
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        
        # Value stream: estimates the value of the state
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        
        # Advantage stream: estimates the advantage of each action
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim)
        )

    def forward(self, features):
        """
        Computes the Q-values from the feature vector.
        
        The Q-value is calculated as V(s) + A(s, a) - mean(A(s, a)).
        """
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values
