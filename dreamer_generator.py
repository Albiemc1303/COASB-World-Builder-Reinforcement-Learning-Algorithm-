# caosb_world_model/modules/dreamer_generator.py

import torch
import torch.nn as nn

class DreamerGenerator(nn.Module):
    """
    A VAE-based generator to produce synthetic, high-value states for
    augmenting the replay buffer.
    
    It learns a latent representation of 'good' states and can sample from
    this latent space to create new, similar states.
    """
    def __init__(self, state_dim, latent_dim):
        super().__init__()
        
        self.vae_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mu, logvar
        )
        
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

    def generate_synthetic(self, high_value_states):
        """
        Generates new synthetic states by encoding and decoding a batch of
        high-value states.
        
        Args:
            high_value_states (torch.Tensor): A batch of high-reward states.
            
        Returns:
            torch.Tensor: The generated synthetic states.
        """
        mu_logvar = self.vae_encoder(high_value_states.float())
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        synthetic = self.vae_decoder(z)
        return synthetic
