# caosb_world_model/modules/ppo_actor_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOActorCritic(nn.Module):
    """
    An Actor-Critic network for PPO. It shares a feature backbone but
    has separate heads for the policy (actor) and value function (critic).
    """
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        
        # Actor network: outputs a probability distribution over actions
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1) # Softmax to get probabilities
        )
        
        # Critic network: outputs the value of the state
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        """
        Computes the action probabilities and state value.
        """
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value
