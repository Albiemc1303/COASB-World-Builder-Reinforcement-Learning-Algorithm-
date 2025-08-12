# CAOSB-World Model Algorithm Implementation
# Phase 1 for LunarLander-v2 Environment
# Requirements: Python 3.8+, PyTorch 2.0+, Gymnasium 0.29+, FAISS, Pandas
# Install if needed: pip install torch gymnasium faiss-cpu pandas

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
from collections import deque, namedtuple
import random
import faiss
import pandas as pd
import pickle
import os
import json
import math

# Hyperparameters
STATE_DIM = 8  # LunarLander state size
ACTION_DIM = 4  # Discrete actions
BUFFER_CAPACITY = 1000000
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor
TAU = 0.005   # Soft update for target network
LR = 0.001    # Learning rate
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
ALPHA_FUN = 0.1  # Fun score weight
ALPHA_ICM = 0.1  # ICM weight
NOISE_STD = 0.1  # Noisy Nets std

# PPO Hyperparameters
PPO_EPOCHS = 4
PPO_CLIP = 0.2
PPO_LAMBDA = 0.95  # GAE lambda

# Experience tuple - UPDATED to include more info for PPO
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

class ExperienceReplayBuffer:
    def __init__(self, capacity, prioritized=False):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.prioritized = prioritized
        if prioritized:
            self.priorities = np.zeros((capacity,), dtype=np.float32)
            self.alpha = 0.6  # PER alpha
            self.beta = 0.4   # PER beta
            self.epsilon = 1e-5

    def push(self, experience, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        if self.prioritized:
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if not self.prioritized:
            if len(self.buffer) < batch_size:
                return []
            return random.sample(self.buffer, batch_size)
        
        # Check if there are enough items to sample from
        if len(self.buffer) < batch_size:
            return [], [], []

        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            if idx < self.capacity:
                self.priorities[idx] = prio + self.epsilon

    def __len__(self):
        return len(self.buffer)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return torch.nn.functional.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return torch.nn.functional.linear(input, self.weight_mu, self.bias_mu)

class DuelingDQNHead(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim)
        )

    def forward(self, features):
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

class PPOActorCritic(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        probs = self.actor(features)
        value = self.critic(features)
        return probs, value

class ICMModule(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU()
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, state, next_state, action):
        phi = self.feature_net(state)
        phi_next = self.feature_net(next_state)
        pred_action = self.inverse_net(torch.cat([phi, phi_next], dim=-1))
        pred_phi_next = self.forward_net(torch.cat([phi, action], dim=-1))
        return pred_action, pred_phi_next, phi_next

class GANRehabModule(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 128),  # +1 for reward
            nn.ReLU(),
            nn.Linear(128, state_dim + action_dim + 1 + state_dim)  # Gen state, action, reward, next_state
        )
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1 + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def generate(self, bad_exp, good_exp):
        # Condition on bad and similar good
        # Simplified for now, just using bad state
        input_tensor = torch.cat([bad_exp.state.float(),
                                  torch.tensor([bad_exp.action]).float(),
                                  torch.tensor([bad_exp.reward]).float()], dim=-1)
        generated = self.generator(input_tensor)
        return generated  # Parse back to exp

class DreamerGenerator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.vae_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mu, logvar
        )
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, STATE_DIM)
        )

    def generate_synthetic(self, high_value_states):
        # VAE-like generation
        mu_logvar = self.vae_encoder(high_value_states)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        synthetic = self.vae_decoder(z)
        return synthetic

class MetaLearner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Wrap base model for MAML-like adaptation

    def adapt(self, experiences):
        # Simple meta-update: fast adapt on positives
        pass  # Implement MAML gradients if needed

class WorldModelBuilder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Transformer accepts (S, N, E) or (N, S, E) where S is sequence length, N is batch, E is feature dim
        # We will use batch_first=True, so (N, S, E)
        encoder_layer = nn.TransformerEncoderLayer(d_model=state_dim, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(state_dim, latent_dim, batch_first=True)  # Context-aware
        self.q_heads = nn.ModuleList([DuelingDQNHead(latent_dim, action_dim) for _ in range(3)])
        self.target_q_heads = nn.ModuleList([DuelingDQNHead(latent_dim, action_dim) for _ in range(3)])
        for head, target in zip(self.q_heads, self.target_q_heads):
            target.load_state_dict(head.state_dict())
            target.eval()
        self.policy = PPOActorCritic(latent_dim, action_dim)
        self.icm = ICMModule(state_dim, action_dim, latent_dim)
        self.gan_rehab = GANRehabModule(state_dim, action_dim)
        self.dreamer = DreamerGenerator(latent_dim)
        self.meta_learner = MetaLearner(self)  # Placeholder
        self.main_buffer = ExperienceReplayBuffer(BUFFER_CAPACITY, prioritized=True)
        self.good_buffer = ExperienceReplayBuffer(BUFFER_CAPACITY // 10)
        self.bad_buffer = ExperienceReplayBuffer(BUFFER_CAPACITY // 10)
        self.policy_buffer = ExperienceReplayBuffer(BUFFER_CAPACITY // 5)
        self.archive = {}  # Model archive
        self.optimizer_q = optim.Adam(self.q_heads.parameters(), lr=LR)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=LR)
        self.optimizer_icm = optim.Adam(self.icm.parameters(), lr=LR)
        self.optimizer_gan = optim.Adam(list(self.gan_rehab.parameters()) + list(self.dreamer.parameters()), lr=LR)
        self.eps = EPS_START
        self.steps = 0
        self.latent_dim = latent_dim
        self.faiss_index = faiss.IndexFlatL2(latent_dim)  # For novelty search
        self.faiss_features = []
        self.fun_history = deque(maxlen=100)  # For fun score
        self.symbolic_rep = {}  # Athena placeholder

    def encode(self, state, hidden=None):
        # NEW: Handle batches correctly
        batch_size = state.size(0)
        state = state.unsqueeze(1) if state.dim() == 1 else state
        # The Transformer needs a (batch_size, seq_len, feature_dim) tensor
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        trans_out = self.encoder(state)
        lstm_out, hidden = self.lstm(trans_out, hidden)
        return lstm_out.squeeze(1), hidden

    def select_action(self, state, hidden=None, explore=True):
        features, hidden = self.encode(state, hidden)
        
        # Reset noise for noisy nets
        for head in self.q_heads:
            for module in head.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

        if explore:
            if random.random() < self.eps:
                return random.randint(0, self.action_dim - 1), hidden
            else:
                with torch.no_grad():
                    q_values = self.q_heads[0](features)  # Standard head for action
                return q_values.argmax().item(), hidden
        else:
            with torch.no_grad():
                probs, value = self.policy(features)
            dist = Categorical(probs)
            return dist.sample().item(), dist.log_prob(torch.tensor([dist.sample().item()])), value, hidden
            
    # NEW: Fun score with FAISS for novelty
    def compute_fun_score(self, reward, episode_rewards, state_features):
        novelty = 0.0
        if self.faiss_index.ntotal > 50:
            # Search for the nearest neighbor
            features_np = state_features.detach().cpu().numpy().reshape(1, -1)
            distances, _ = self.faiss_index.search(features_np, 1)
            # Inverse of the distance as a novelty score
            novelty = 1.0 / (distances[0][0] + 1e-5)
        else:
            # High novelty for the first few states
            novelty = 1.0

        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        streak = 1 if reward > avg_reward else 0
        mastery = 1 if reward > max(episode_rewards or [0]) else 0
        if len(self.fun_history) > 5 and np.mean(self.fun_history[-5:]) < 0 and reward > 0:
            recovery = 1
        else:
            recovery = 0
        
        fun = 0.2 * novelty + 0.3 * streak + 0.3 * mastery + 0.2 * recovery
        dopamine = 0.01 + math.tanh(fun) + (1 if reward > 10 else 0) * 0.5
        self.fun_history.append(reward)
        return fun * dopamine

    # NEW: Update all Q-heads
    def update_q(self, batch, head_idx):
        states, actions, rewards, next_states, dones, _, _ = zip(*batch)
        states = torch.stack([torch.from_numpy(s) for s in states]).float()
        next_states = torch.stack([torch.from_numpy(s) for s in next_states]).float()
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()

        features, _ = self.encode(states)
        next_features, _ = self.encode(next_states)

        q_values = self.q_heads[head_idx](features).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Double DQN
            next_q_values = self.q_heads[head_idx](next_features)
            next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
            next_q_target = self.target_q_heads[head_idx](next_features).gather(1, next_actions).squeeze(1)
            target = rewards + GAMMA * next_q_target * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()

        # Soft update target
        for param, target_param in zip(self.q_heads[head_idx].parameters(), self.target_q_heads[head_idx].parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return loss.item(), (target - q_values).abs().detach().cpu().numpy()  # For priorities

    # NEW: Full PPO update implementation
    def update_policy(self):
        if len(self.policy_buffer) < BATCH_SIZE:
            return 0
        
        states, actions, rewards, next_states, dones, log_probs, values = zip(*self.policy_buffer.buffer)
        states = torch.stack([torch.from_numpy(s) for s in states]).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        old_log_probs = torch.tensor(log_probs).float()
        old_values = torch.tensor(values).float()
        
        # Calculate returns and advantages using GAE
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0
        with torch.no_grad():
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_state_tensor = torch.from_numpy(next_states[t]).float().unsqueeze(0)
                    next_features, _ = self.encode(next_state_tensor)
                    _, next_value = self.policy(next_features)
                    next_value = next_value.squeeze(0)
                
                features_t, _ = self.encode(states[t].unsqueeze(0))
                _, value_t = self.policy(features_t)
                value_t = value_t.squeeze(0)

                delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - value_t
                last_gae_lambda = delta + GAMMA * PPO_LAMBDA * (1 - dones[t]) * last_gae_lambda
                returns[t] = last_gae_lambda + value_t
        
        # Normalize advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO training loop
        for _ in range(PPO_EPOCHS):
            batch_indices = np.random.choice(len(self.policy_buffer), BATCH_SIZE, replace=False)
            
            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_advantages = advantages[batch_indices]
            b_returns = returns[batch_indices]
            b_old_log_probs = old_log_probs[batch_indices]

            b_features, _ = self.encode(b_states)
            probs, values = self.policy(b_features)
            dist = Categorical(probs)
            log_probs = dist.log_prob(b_actions)
            
            # Policy loss
            ratio = torch.exp(log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(-1), b_returns)
            
            # Total loss and update
            loss = policy_loss + 0.5 * value_loss
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

        self.policy_buffer.buffer.clear()
        return loss.item()

    def update_icm(self, batch):
        states, actions, _, next_states, _, _, _ = zip(*batch)
        states = torch.stack([torch.from_numpy(s) for s in states]).float()
        next_states = torch.stack([torch.from_numpy(s) for s in next_states]).float()
        actions = torch.tensor(actions).long()
        one_hot_actions = torch.eye(self.action_dim)[actions]

        pred_actions, pred_phi_next, phi_next = self.icm(states, next_states, one_hot_actions)
        inverse_loss = nn.CrossEntropyLoss()(pred_actions, actions)
        forward_loss = 0.5 * ((pred_phi_next - phi_next.detach()) ** 2).mean()
        loss = inverse_loss + forward_loss
        self.optimizer_icm.zero_grad()
        loss.backward()
        self.optimizer_icm.step()
        intrinsic_reward = forward_loss.item()  # Prediction error as reward
        return intrinsic_reward

    def rehab_experiences(self):
        if len(self.bad_buffer) < 1 or len(self.good_buffer) < 1 or self.faiss_index.ntotal == 0:
            return
        
        # Sample bad experiences
        bad_exp = random.choice(self.bad_buffer.buffer)
        
        # FAISS search for similar good
        bad_feat, _ = self.encode(torch.tensor(bad_exp.state).float().unsqueeze(0))
        bad_feat = bad_feat.detach().cpu().numpy().reshape(1, -1)
        
        distances, idx = self.faiss_index.search(bad_feat, 1)
        
        # Find the corresponding good experience in the good_buffer
        good_exp = None
        for exp in self.good_buffer.buffer:
            feat, _ = self.encode(torch.tensor(exp.state).float().unsqueeze(0))
            feat = feat.detach().cpu().numpy().reshape(1, -1)
            if np.allclose(feat, self.faiss_features[idx[0][0]]):
                good_exp = exp
                break
        
        if good_exp is None:
            return

        # Simplified GAN generation and training
        generated = self.gan_rehab.generate(bad_exp, good_exp)
        
        # Train GAN
        real_state = torch.tensor(good_exp.state).float()
        real_action = torch.tensor([good_exp.action]).float()
        real_reward = torch.tensor([good_exp.reward]).float()
        real_next_state = torch.tensor(good_exp.next_state).float()
        real = torch.cat([real_state, real_action, real_reward, real_next_state])
        
        fake = generated
        
        d_real = self.gan_rehab.discriminator(real.detach())
        d_fake = self.gan_rehab.discriminator(fake.detach())
        
        # Discriminator Loss
        loss_d = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
        
        # Generator Loss
        d_fake_g = self.gan_rehab.discriminator(fake)
        loss_g = -torch.mean(torch.log(d_fake_g))
        
        self.optimizer_gan.zero_grad()
        loss_d.backward()
        loss_g.backward()
        self.optimizer_gan.step()

        # Push rehabbed to main if good
        if d_fake_g > 0.5:
            # Correctly parse the generated experience
            generated_state = generated[0:STATE_DIM]
            generated_action = generated[STATE_DIM:STATE_DIM+1]
            generated_reward = generated[STATE_DIM+1:STATE_DIM+2]
            generated_next_state = generated[STATE_DIM+2:STATE_DIM+2+STATE_DIM]

            rehab_exp = Experience(generated_state.detach().numpy(),
                                   int(generated_action.round().item()),
                                   generated_reward.item(),
                                   generated_next_state.detach().numpy(),
                                   False, None, None)
            self.main_buffer.push(rehab_exp)

    def dream_generate(self):
        if len(self.good_buffer) < 1:
            return
        high_states_list = random.sample(self.good_buffer.buffer, min(10, len(self.good_buffer)))
        high_states = torch.stack([torch.tensor(exp.state) for exp in high_states_list])
        synthetic_states = self.dreamer.generate_synthetic(high_states.float())
        
        for s in synthetic_states:
            fake_exp = Experience(s.detach().numpy(), random.randint(0,3), 10, s.detach().numpy(), False, None, None)
            self.main_buffer.push(fake_exp, priority=2.0)

    def sync_buffers(self):
        for exp in self.good_buffer.buffer:
            self.main_buffer.push(exp, priority=2.0)
        for exp in self.bad_buffer.buffer:
            self.main_buffer.push(exp, priority=0.5)
        self.good_buffer.buffer.clear()
        self.bad_buffer.buffer.clear()

    def train_step(self, env, hidden=None):
        state, _ = env.reset()
        state = torch.tensor(state).float()
        episode_reward = 0
        episode_rewards = []
        done = False
        step_count = 0
        
        while not done and step_count < 500: # Add step limit for safety
            # Select action with PPO for exploration
            with torch.no_grad():
                features, hidden = self.encode(state.unsqueeze(0), hidden)
                probs, value = self.policy(features)
                dist = Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
                value = value.item()

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state).float()
            done = done or truncated

            # Fun score with novelty from FAISS
            state_features, _ = self.encode(state.unsqueeze(0))
            fun_score = self.compute_fun_score(reward, episode_rewards, state_features)
            
            # Update FAISS index incrementally
            features_np = state_features.detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add(features_np)
            self.faiss_features.append(features_np)
            
            total_reward = reward + ALPHA_FUN * fun_score

            icm_reward = self.update_icm([(state.numpy(), action, reward, next_state.numpy(), done, None, None)])
            total_reward += ALPHA_ICM * icm_reward

            exp = Experience(state.numpy(), action, total_reward, next_state.numpy(), done, log_prob, value)

            q_delta = 0  # Placeholder, needs to be calculated for real
            # A simple way to get q_delta is to check if the reward is positive
            if reward > 0:
                self.good_buffer.push(exp)
            elif reward < 0:
                self.bad_buffer.push(exp)
            else:
                self.main_buffer.push(exp)

            self.policy_buffer.push(exp)

            # NEW: Update all Q-heads
            for idx in range(3):
                if len(self.main_buffer) > BATCH_SIZE:
                    batch, indices, weights = self.main_buffer.sample(BATCH_SIZE)
                    if batch: # Check if batch is not empty
                        loss, td_errors = self.update_q(batch, idx)
                        self.main_buffer.update_priorities(indices, td_errors)
            
            state = next_state
            episode_reward += reward
            episode_rewards.append(reward)
            self.steps += 1
            self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
            step_count += 1
            
        self.sync_buffers()
        policy_loss = self.update_policy()
        self.rehab_experiences()
        self.dream_generate()

        return {'episode_reward': episode_reward, 'policy_loss': policy_loss}

    def athena_extract(self, env):
        self.symbolic_rep = {
            'dynamics': 'y_{t+1} = y_t + vy_t * dt - 0.5 * gravity * dt^2',
            'goal': 'land on pad with |x|<0.1, |vy|<0.3, legs grounded',
            'gravity': -10.0
        }

    def save_pics(self, path='checkpoint.pth'):
        state = {
            'model_state': self.state_dict(),
            'optimizer_q': self.optimizer_q.state_dict(),
            'buffers': {
                'main': pickle.dumps(self.main_buffer),
                'good': pickle.dumps(self.good_buffer),
                'bad': pickle.dumps(self.bad_buffer),
                'policy': pickle.dumps(self.policy_buffer)
            },
            'archive': self.archive,
            'steps': self.steps,
            'symbolic': self.symbolic_rep,
            'faiss_features': self.faiss_features
        }
        torch.save(state, path)

    def load_pics(self, path='checkpoint.pth'):
        if os.path.exists(path):
            state = torch.load(path)
            self.load_state_dict(state['model_state'])
            self.optimizer_q.load_state_dict(state['optimizer_q'])
            self.main_buffer = pickle.loads(state['buffers']['main'])
            self.good_buffer = pickle.loads(state['buffers']['good'])
            self.bad_buffer = pickle.loads(state['buffers']['bad'])
            self.policy_buffer = pickle.loads(state['buffers']['policy'])
            self.archive = state['archive']
            self.steps = state['steps']
            self.symbolic_rep = state['symbolic']
            self.faiss_features = state['faiss_features']
            
            # Rebuild FAISS index on load
            self.faiss_index = faiss.IndexFlatL2(self.latent_dim)
            if self.faiss_features:
                self.faiss_index.add(np.array(self.faiss_features).astype('float32'))


class ShapedLunarLander(gym.Wrapper):
    def __init__(self, env, stage=1):
        super().__init__(env)
        self.stage = stage

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        x, y, vx, vy, theta, omega, leg1, leg2 = obs
        shaped = 0
        if self.stage == 1:
            shaped += 5 if abs(vy) < 0.1 and y > 0.5 else -1
        elif self.stage == 2:
            shaped += 3 if vy < 0 and abs(vx) < 0.2 else -1
        elif self.stage == 3:
            shaped += 10 if leg1 > 0 and leg2 > 0 and abs(vy) < 0.3 else -5
        shaped -= abs(theta) * 0.5
        return obs, reward + shaped, done, truncated, info

# Training
if __name__ == '__main__':
    env = ShapedLunarLander(gym.make("LunarLander-v2"))
    agent = WorldModelBuilder(STATE_DIM, ACTION_DIM)
    agent.athena_extract(env)
    log_df = pd.DataFrame(columns=['episode', 'reward'])
    telemetry = []

    for episode in range(1000):
        metrics = agent.train_step(env)
        log_df = pd.concat([log_df, pd.DataFrame({'episode': [episode], 'reward': [metrics['episode_reward']]})], ignore_index=True)
        log_df.to_csv('training_logs.csv', index=False)
        print(f"Episode {episode}: Reward {metrics['episode_reward']}")
        
        telemetry.append({'episode': episode, 'q_loss': 0, 'policy_loss': metrics['policy_loss']})
        pd.DataFrame(telemetry).to_csv('telemetry.csv', index=False)
        
        if episode % 100 == 0:
            agent.save_pics(f'checkpoint_{episode}.pth')
            agent.archive[f'model_{episode}'] = agent.state_dict()

        if metrics['episode_reward'] > 100 and env.stage < 3:
            env.stage += 1
            print(f"Advancing to curriculum stage {env.stage}")

    env.close()

