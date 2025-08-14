# caosb_world_model/modules/gan_rehab.py

import torch
import torch.nn as nn
from experience import Experience

class WGANRehabModule(nn.Module):
    """
    A Wasserstein Generative Adversarial Network (WGAN) for rehabilitating 'bad' experiences.
    
    The generator learns to produce 'good' experiences conditioned on 'bad' ones.
    The critic estimates the Wasserstein distance between real and generated distributions.
    This architecture provides more stable training and prevents mode collapse.
    """
    def __init__(self, state_dim, action_dim, critic_iters=5):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_iters = critic_iters
        
        # Generator: Generates a new experience (s, a, r, s') from a given experience
        # Input size: state_dim + action_dim + 1 (for reward)
        # Output size: state_dim + action_dim + 1 + state_dim (for s, a, r, s')
        self.generator = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim + action_dim + 1 + state_dim)
        )
        
        # Critic: Estimates the Wasserstein distance. No sigmoid in the output.
        # Input size: state_dim + action_dim + 1 + state_dim
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1 + state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Optimizer for the critic. We'll use a separate one as per WGAN paper.
        self.optimizer_critic = None

    def forward(self, x):
        """Passes an input through the critic."""
        return self.critic(x)

    def generate(self, bad_exp):
        """
        Generates a synthetic experience from a 'bad' experience.
        
        Args:
            bad_exp (Experience): The 'bad' experience to condition on.
            
        Returns:
            torch.Tensor: The generated experience tensor.
        """
        input_tensor = torch.cat([
            torch.from_numpy(bad_exp.state).float(),
            torch.tensor([bad_exp.action]).float(),
            torch.tensor([bad_exp.reward]).float()
        ], dim=-1).unsqueeze(0)
        
        generated = self.generator(input_tensor)
        return generated.squeeze(0)
