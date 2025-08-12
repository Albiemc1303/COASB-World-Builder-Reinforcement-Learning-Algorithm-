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
from caosb_world_model.modules.gan_rehab import WGANRehabModule
from caosb_world_model.modules.dreamer_generator import DreamerGenerator
from caosb_world_model.agents.meta_learner import MetaLearner
from caosb_world_model.modules.noisy_layers import NoisyLinear

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
        self.batch_size = hyperparameters['batch_size']
        self.eps_start = hyperparameters['eps_start']
        self.eps_end = hyperparameters['eps_end']
        self.eps_decay = hyperparameters['eps_decay']
        
        # New WGAN hyperparameters
        self.gan_lr = hyperparameters.get('gan_lr', 1e-4)
        self.critic_iters = hyperparameters.get('critic_iters', 5)

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
        self.gan_rehab = WGANRehabModule(state_dim, action_dim, self.critic_iters)
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
        self.optimizer_gan = optim.Adam(self.gan_rehab.generator.parameters(), lr=self.gan_lr)
        self.optimizer_critic = optim.Adam(self.gan_rehab.critic.parameters(), lr=self.gan_lr)

        # Training State
        self.eps = self.eps_start
        self.steps = 0
        self.archive = {}  # Model archive
        self.faiss_index = faiss.IndexFlatL2(self.latent_dim)
        self.faiss_features = []
        self.fun_history = deque(maxlen=100)
        
        # New: Symbolic knowledge base, a dictionary to store learned rules
        self.symbolic_knowledge_base = {
            'rules': [], # A list to hold our learned symbolic rules
            'dynamics': {},
            'goals': {}
        }


    def encode(self, state, hidden=None):
        """Encodes the state using a Transformer-LSTM model."""
        batch_size = state.size(0)
        state = state.unsqueeze(1) if state.dim() == 1 else state
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        trans_out = self.encoder(state)
        lstm_out, hidden = self.lstm(trans_out, hidden)
        return lstm_out.squeeze(1), hidden

    def get_ppo_action(self, state, hidden=None):
        """
        Selects an action using the PPO policy and returns all necessary data
        for training.
        """
        features, hidden = self.encode(state, hidden)
        with torch.no_grad():
            probs, value = self.policy(features)
        dist = Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=probs.device))
        return action, log_prob, value.squeeze(0), hidden

    def select_action(self, state, hidden=None):
        """
        Selects an action greedily using the first Dueling DQN head for
        evaluation purposes.
        """
        features, hidden = self.encode(state, hidden)
        
        for head in self.q_heads:
            for module in head.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

        with torch.no_grad():
            q_values = self.q_heads[0](features)
        action = q_values.argmax().item()
        return action, hidden

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
        dopamine = 0.01 + math.tanh(fun) + (1 if reward > 10 else 0) * 0.5
        self.fun_history.append(reward)
        return fun * dopamine

    def update_q(self, batch, head_idx):
        """Updates one of the Q-networks using Double DQN and PER."""
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
            next_q_values = self.q_heads[head_idx](next_features)
            next_actions = next_q_values.argmax(dim=1).unsqueeze(1)
            next_q_target = self.target_q_heads[head_idx](next_features).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q_target * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer_q.zero_grad()
        loss.backward()
        self.optimizer_q.step()

        for param, target_param in zip(self.q_heads[head_idx].parameters(), self.target_q_heads[head_idx].parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item(), (target - q_values).abs().detach().cpu().numpy()

    def update_policy(self):
        """Performs a full PPO update on the policy buffer."""
        if len(self.policy_buffer) < self.batch_size:
            return 0
        
        states, actions, rewards, next_states, dones, log_probs, values = zip(*self.policy_buffer.buffer)
        states = torch.stack([torch.from_numpy(s) for s in states]).float()
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).float()
        old_log_probs = torch.tensor(log_probs).float()
        old_values = torch.tensor(values).float()
        
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0.0
        
        with torch.no_grad():
            # Correct and more standard GAE calculation
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = torch.tensor(0.0)
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = old_values[t+1]
                
                delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - old_values[t]
                last_gae_lambda = delta + self.gamma * self.ppo_lambda * next_non_terminal * last_gae_lambda
                advantages[t] = last_gae_lambda
                returns[t] = advantages[t] + old_values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            batch_indices = np.random.choice(len(self.policy_buffer), self.batch_size, replace=False)
            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_advantages = advantages[batch_indices]
            b_returns = returns[batch_indices]
            b_old_log_probs = old_log_probs[batch_indices]

            b_features, _ = self.encode(b_states)
            probs, values = self.policy(b_features)
            dist = Categorical(probs)
            log_probs = dist.log_prob(b_actions)
            
            ratio = torch.exp(log_probs - b_old_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(-1), b_returns)
            
            loss = policy_loss + 0.5 * value_loss
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

        self.policy_buffer.buffer.clear()
        return loss.item()

    def update_icm(self, batch):
        """Updates the ICM and returns the intrinsic reward."""
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
        intrinsic_reward = forward_loss.item()
        return intrinsic_reward

    def rehab_experiences(self):
        """Rehabilitates 'bad' experiences using the WGAN."""
        if len(self.bad_buffer) < 1 or len(self.good_buffer) < self.batch_size or self.faiss_index.ntotal == 0:
            return
        
        bad_exp = random.choice(self.bad_buffer.buffer)
        bad_feat, _ = self.encode(torch.tensor(bad_exp.state).float().unsqueeze(0))
        bad_feat_np = bad_feat.detach().cpu().numpy().reshape(1, -1)
        
        distances, idx = self.faiss_index.search(bad_feat_np, 1)
        
        good_exp = None
        for exp in self.good_buffer.buffer:
            feat, _ = self.encode(torch.tensor(exp.state).float().unsqueeze(0))
            feat = feat.detach().cpu().numpy().reshape(1, -1)
            if np.allclose(feat, self.faiss_features[idx[0][0]]):
                good_exp = exp
                break
        
        if good_exp is None:
            return

        # WGAN training loop: Train the critic multiple times per generator update
        for _ in range(self.critic_iters):
            self.optimizer_critic.zero_grad()
            
            # Get real and fake experiences
            real_state = torch.tensor(good_exp.state).float()
            real_action = torch.tensor([good_exp.action]).float()
            real_reward = torch.tensor([good_exp.reward]).float()
            real_next_state = torch.tensor(good_exp.next_state).float()
            real = torch.cat([real_state, real_action, real_reward, real_next_state]).unsqueeze(0)
            
            generated = self.gan_rehab.generate(bad_exp).unsqueeze(0)

            # Calculate critic losses
            d_real = self.gan_rehab.critic(real.detach())
            d_fake = self.gan_rehab.critic(generated.detach())
            
            loss_d = d_fake.mean() - d_real.mean()
            
            # Backpropagate and clip weights
            loss_d.backward()
            self.optimizer_critic.step()
            for p in self.gan_rehab.critic.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Train the generator
        self.optimizer_gan.zero_grad()
        generated = self.gan_rehab.generate(bad_exp).unsqueeze(0)
        d_fake_g = self.gan_rehab.critic(generated)
        loss_g = -d_fake_g.mean()
        loss_g.backward()
        self.optimizer_gan.step()

        if d_fake_g.mean().item() > d_real.mean().item() * 0.9:
            gen_exp_data = generated.squeeze(0).detach().cpu().numpy()
            gen_s = gen_exp_data[0:self.state_dim]
            gen_a = int(gen_exp_data[self.state_dim].round())
            gen_r = gen_exp_data[self.state_dim+1]
            gen_ns = gen_exp_data[self.state_dim+2:]

            rehab_exp = Experience(gen_s, gen_a, gen_r, gen_ns, False, None, None)
            self.main_buffer.push(rehab_exp)

    def dream_generate(self):
        """Generates synthetic states using the Dreamer VAE."""
        if len(self.good_buffer) < 1:
            return
        high_states_list = random.sample(self.good_buffer.buffer, min(10, len(self.good_buffer)))
        high_states = torch.stack([torch.tensor(exp.state) for exp in high_states_list])
        synthetic_states = self.dreamer.generate_synthetic(high_states.float())
        
        for s in synthetic_states:
            fake_exp = Experience(s.detach().numpy(), random.randint(0,3), 10, s.detach().numpy(), False, None, None)
            self.main_buffer.push(fake_exp, priority=2.0)

    def sync_buffers(self):
        """Synchronizes experiences from good/bad buffers to the main buffer."""
        for exp in self.good_buffer.buffer:
            self.main_buffer.push(exp, priority=2.0)
        for exp in self.bad_buffer.buffer:
            self.main_buffer.push(exp, priority=0.5)
        self.good_buffer.buffer.clear()
        self.bad_buffer.buffer.clear()

    def athena_extract(self):
        """
        Analyzes experiences to extract and store symbolic, causal rules.
        This is an initial, rule-based approach for knowledge extraction.
        """
        if len(self.good_buffer) < 1 or len(self.bad_buffer) < 1:
            return

        # Analyze good experiences to infer successful patterns
        good_states = np.array([exp.state for exp in self.good_buffer.buffer])
        good_actions = np.array([exp.action for exp in self.good_buffer.buffer])
        good_next_states = np.array([exp.next_state for exp in self.good_buffer.buffer])
        
        if good_states.shape[0] > 10:
            y_pos_change = good_next_states[:, 1] - good_states[:, 1]
            thrust_actions = good_actions == 2
            
            if np.mean(y_pos_change[thrust_actions]) > 0.05:
                rule = {
                    'if': {'action': 2, 'effect': 'positive_y_velocity'},
                    'then': {'outcome': 'desirable', 'reward_bonus': 0.5}
                }
                if rule not in self.symbolic_knowledge_base['rules']:
                    self.symbolic_knowledge_base['rules'].append(rule)

        # Analyze bad experiences to infer failure conditions
        bad_states = np.array([exp.state for exp in self.bad_buffer.buffer])
        
        if bad_states.shape[0] > 10:
            angle_too_large = np.abs(bad_states[:, 4]) > 0.5
            
            if np.mean(angle_too_large) > 0.5:
                rule = {
                    'if': {'state': 'angle_too_large'},
                    'then': {'outcome': 'undesirable', 'reward_penalty': -1.0}
                }
                if rule not in self.symbolic_knowledge_base['rules']:
                    self.symbolic_knowledge_base['rules'].append(rule)

    def apply_symbolic_reward(self, current_state, current_action):
        """
        Uses the learned symbolic knowledge to provide an additional reward signal.
        """
        symbolic_reward = 0.0
        for rule in self.symbolic_knowledge_base['rules']:
            # Check for action-based rules
            if 'action' in rule['if'] and rule['if']['action'] == current_action:
                if rule['then']['outcome'] == 'desirable':
                    symbolic_reward += rule['then']['reward_bonus']
                elif rule['then']['outcome'] == 'undesirable':
                    symbolic_reward += rule['then']['reward_penalty']
        
        # Add state-based rules
        for rule in self.symbolic_knowledge_base['rules']:
            if 'state' in rule['if'] and rule['if']['state'] == 'angle_too_large':
                if np.abs(current_state[4]) > 0.5:
                    symbolic_reward += rule['then']['reward_penalty']
                    
        return symbolic_reward

    def train_step(self, env):
        """Performs a single episode of training."""
        state, _ = env.reset()
        state = torch.tensor(state).float()
        episode_reward = 0
        episode_rewards = []
        done = False
        step_count = 0
        
        hidden = None
        
        while not done and step_count < 500:
            # New: Call athena_extract periodically to learn new rules
            if self.steps % 500 == 0 and self.steps > 0:
                self.athena_extract()
            
            action, log_prob, value, hidden = self.get_ppo_action(state.unsqueeze(0), hidden=hidden)
            
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = torch.tensor(next_state).float()
            done = done or truncated

            state_features, _ = self.encode(state.unsqueeze(0))
            fun_score = self.compute_fun_score(reward, episode_rewards, state_features)
            
            features_np = state_features.detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add(features_np)
            self.faiss_features.append(features_np)
            
            # New: Use symbolic knowledge to shape the reward
            symbolic_reward = self.apply_symbolic_reward(state.numpy(), action)
            total_reward = reward + self.alpha_fun * fun_score + symbolic_reward
            
            icm_reward = self.update_icm([(state.numpy(), action, reward, next_state.numpy(), done, None, None)])
            total_reward += self.alpha_icm * icm_reward

            exp = Experience(state.numpy(), action, total_reward, next_state.numpy(), done, log_prob, value)

            if reward > 0:
                self.good_buffer.push(exp)
            elif reward < 0:
                self.bad_buffer.push(exp)
            else:
                self.main_buffer.push(exp)

            self.policy_buffer.push(exp)

            for idx in range(3):
                if len(self.main_buffer) > self.batch_size:
                    batch, indices, weights = self.main_buffer.sample(self.batch_size)
                    if batch:
                        loss, td_errors = self.update_q(batch, idx)
                        self.main_buffer.update_priorities(indices, td_errors)
            
            state = next_state
            episode_reward += reward
            episode_rewards.append(reward)
            self.steps += 1
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
            step_count += 1
            
        self.sync_buffers()
        policy_loss = self.update_policy()
        self.rehab_experiences()
        self.dream_generate()

        return {'episode_reward': episode_reward, 'policy_loss': policy_loss}

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
            'symbolic_knowledge_base': self.symbolic_knowledge_base,
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
            self.symbolic_knowledge_base = state.get('symbolic_knowledge_base', {
                'rules': [], 'dynamics': {}, 'goals': {}
            })
            self.faiss_features = state['faiss_features']
            
            self.faiss_index = faiss.IndexFlatL2(self.latent_dim)
            if self.faiss_features:
                faiss_features_np = np.array(self.faiss_features).astype('float32').reshape(-1, self.latent_dim)
                self.faiss_index.add(faiss_features_np)
