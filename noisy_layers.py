# caosb_world_model/modules/noisy_layers.py

import torch
import torch.nn as nn
import math

class NoisyLinear(nn.Module):
    """
    A NoisyLinear layer for adding exploration noise to network weights.
    
    This replaces epsilon-greedy exploration with a learnable noise injection
    that is added to the linear layer's weights.
    """
    def __init__(self, in_features, out_features, std_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Parameters for mean and variance of the weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initializes the parameters of the layer."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Generates new noise vectors for the weights and biases."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Helper function to create a scaled noise vector."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        """
        Forward pass. During training, noise is added. During evaluation,
        only the mean weights are used.
        """
        if self.training:
            return torch.nn.functional.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return torch.nn.functional.linear(input, self.weight_mu, self.bias_mu)
