# caosb_world_model/agents/world_model_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
import faiss
import math
import pickle
import os
import pandas as pd

# Import all the refactored modules
from caosb_world_model.core.buffers import ExperienceReplayBuffer
from caosb_world_model.core.experience import Experience
from caosb_world_model.modules.dueling_dqn_head import DuelingDQNHead
from caosb_world_model.modules.ppo_actor_critic import PPOActorCritic
from caosb_world_model.modules.icm import ICMModule
from caosb_world_model.modules.gan_rehab import GANRehabModule
from caosb_world_model.modules.dreamer_generator import DreamerGenerator
from caosb_world_model.agents.meta_learner import MetaLearner

class WorldModelBuilder(nn.Module):
    """
    The main World Model agent, which orchestrates all the sub-modules
    to perform curriculum-based reinforcement learning.

    This class handles state encoding, action selection, training updates for
    DQN, PPO, ICM, GAN, and manages the various replay buffers.
    """
    def __init__(self, state_dim, action_dim, hyperparameters):
        super().__init__()
        
        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = hyperparameters['latent_dim']
        self.gamma = hyperparameters['gamma']
        self.tau = hyperparameters['tau']
        self.lr = hyperparameters['lr']
        self.alpha_fun = hyperparameters['alpha_fun']
        self.alpha_icm = hyperparameters['alpha_icm']
        self.ppo_epochs = hyperparameters['ppo_epochs']
        self.ppo_clip = hyperparameters['ppo_clip']
        self.ppo_lambda = hyperparameters['ppo_lambda']
        
        # Core Networks
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer_encoder, num_layers=2)
        self.lstm = nn.LSTM(state_dim, self.latent_dim, batch_first=True)

        # Learning Components
        self.q_heads = nn.ModuleList([DuelingDQNHead(self.latent_dim, action_dim) for _ in range(3)])
        self.target_q_heads = nn.ModuleList([DuelingDQNHead(self.latent_dim, action_dim) for _ in range(3)])
        for head, target in zip(self.q_heads, self.target_q_heads):
            target.load_state_dict(head.state_dict())
            target.eval()
            
        self.policy = PPOActorCritic(self.latent_dim, action_dim)
        self.icm = ICMModule(state_dim, action_dim, self.latent_dim)
        self.gan_rehab = GANRehabModule(state_dim, action_dim)
        self.dreamer = DreamerGenerator(state_dim, self.latent_dim)
        self.meta_learner = MetaLearner(self)

        # Buffers
        self.main_buffer = ExperienceReplayBuffer(hyperparameters['buffer_capacity'], prioritized=True)
        self.good_buffer = ExperienceReplayBuffer(hyperparameters['buffer_capacity'] // 10)
        self.bad_buffer = ExperienceReplayBuffer(hyperparameters['buffer_capacity'] // 10)
        self.policy_buffer = ExperienceReplayBuffer(hyperparameters['buffer_capacity'] // 5)

        # Optimizers
        self.optimizer_q = optim.Adam(self.q_heads.parameters(), lr=self.lr)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_icm = optim.Adam(self.icm.parameters(), lr=self.lr)
        self.optimizer_gan = optim.Adam(
            list(self.gan_rehab.parameters()) + list(self.dreamer.parameters()), lr=self.lr
        )

        # Training State
        self.eps = hyperparameters['eps_start']
        self.eps_end = hyperparameters['eps_end']
        self.eps_decay = hyperparameters['eps_decay']
        self.steps = 0
        self.archive = {}  # Model archive
        self.faiss_index = faiss.IndexFlatL2(self.latent_dim)
        self.faiss_features = []
        self.fun_history = deque(maxlen=100)
        self.symbolic_rep = {}

    def encode(self, state, hidden=None):
        """Encodes the state using a Transformer-LSTM model."""
        batch_size = state.size(0)
        state = state.unsqueeze(1) if state.dim() == 1 else state
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        trans_out = self.encoder(state)
        lstm_out, hidden = self.lstm(trans_out, hidden)
        return lstm_out.squeeze(1), hidden

    def select_action(self, state, hidden=None, explore=True):
        """Selects an action using either epsilon-greedy or PPO-based exploration."""
        features, hidden = self.encode(state, hidden)
        
        if explore:
            # Reset noise for noisy nets
            for head in self.q_heads:
                for module in head.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()

            if random.random() < self.eps:
                return random.randint(0, self.action_dim - 1), None, None, hidden
            else:
                with torch.no_grad():
                    q_values = self.q_heads[0](features)
                action = q_values.argmax().item()
                return action, None, None, hidden
        else:
            with torch.no_grad():
                probs, value = self.policy(features)
            dist = Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))
            return action, log_prob, value.squeeze(0), hidden

    def compute_fun_score(self, reward, episode_rewards, state_features):
        """Calculates a fun score based on novelty, streaks, and mastery."""
        novelty = 0.0
        if self.faiss_index.ntotal > 50:
            features_np = state_features.detach().cpu().numpy().reshape(1, -1)
            distances, _ = self.faiss_index.search(features_np, 1)
            novelty = 1.0 / (distances[0][0] + 1e-5)
        else:
            novelty = 1.0

        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        streak = 1 if reward > avg_reward else 0
        mastery = 1 if reward > max(episode_rewards or [0]) else 0
        recovery = 1 if len(self.fun_history) > 5 and np.mean(self.fun_history[-5:]) < 0 and reward > 0 else 0
        
        fun = 0.2 * novelty + 0.3 * streak + 0.3 * mastery + 0.2 * recovery
        dopamine = 0.01 + math.tanh(fun) + (1 if reward > 10 else 0) *
