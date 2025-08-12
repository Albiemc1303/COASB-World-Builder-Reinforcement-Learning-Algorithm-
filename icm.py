# caosb_world_model/modules/icm.py

import torch
import torch.nn as nn

class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for generating an intrinsic reward signal.
    
    The module predicts the next state's feature representation and the action
    taken, and the forward model prediction error serves as an intrinsic reward.
    """
    def __init__(self, state_dim, action_dim, feature_dim):
        super().__init__()
        
        # Feature network: maps a state to a feature representation
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU()
        )
        
        # Inverse network: predicts the action from two consecutive states
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Forward network: predicts the next state's feature representation
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, state, next_state, action):
        """
        Performs a forward pass through the ICM.
        
        Args:
            state (torch.Tensor): The current state.
            next_state (torch.Tensor): The next state.
            action (torch.Tensor): The one-hot encoded action.
            
        Returns:
            tuple: A tuple containing the predicted action, predicted next feature,
                   and the actual next feature.
        """
        phi = self.feature_net(state)
        phi_next = self.feature_net(next_state)
        
        pred_action = self.inverse_net(torch.cat([phi, phi_next], dim=-1))
        
        pred_phi_next = self.forward_net(torch.cat([phi, action], dim=-1))
        
        return pred_action, pred_phi_next, phi_next
