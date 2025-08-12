# caosb_world_model/modules/gan_rehab.py

import torch
import torch.nn as nn
from caosb_world_model.core.experience import Experience

class GANRehabModule(nn.Module):
    """
    A Generative Adversarial Network (GAN) for rehabilitating 'bad' experiences.
    
    The generator learns to produce 'good' experiences conditioned on 'bad' ones.
    The discriminator learns to distinguish between real 'good' experiences
    and generated 'good' experiences.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Generator: Generates a new experience (s, a, r, s') from a given experience
        # Input size: state_dim + action_dim + 1 (for reward)
        # Output size: state_dim + action_dim + 1 + state_dim (for s, a, r, s')
        self.generator = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim + action_dim + 1 + state_dim)
        )
        
        # Discriminator: Predicts if an experience is real or fake
        # Input size: state_dim + action_dim + 1 + state_dim
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1 + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Passes an input through the discriminator."""
        return self.discriminator(x)

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
        ], dim=-1).unsqueeze(0) # Add batch dimension
        
        generated = self.generator(input_tensor)
        return generated.squeeze(0)
