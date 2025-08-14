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
import copy
import gymnasium as gym

# Import all the refactored modules
from buffers import ExperienceReplayBuffer
from experience import Experience
from dueling_dqn_head import DuelingDQNHead
from ppo_actor_critic import PPOActorCritic
from icm import ICMModule
from gan_rehab import WGANRehabModule
from dreamer_generator import DreamerGenerator
from noisy_layers import NoisyLinear
from symbolic_planner import SymbolicPlanner

class WorldModelBuilder(nn.Module):
    """
    The main World Model agent, which orchestrates all the sub-modules
    to perform curriculum-based reinforcement learning.

    This class handles state encoding, action selection, training updates for
    DQN, PPO, ICM, GAN, and manages the various replay buffers. It now
    also integrates a SymbolicPlanner for high-level goal-oriented control.
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
        
        # New MAML hyperparameters
        self.meta_lr = hyperparameters.get('meta_lr', 1e-5)
        self.num_inner_updates = hyperparameters.get('num_inner_updates', 1)
        self.inner_lr = hyperparameters.get('inner_lr', 1e-3)
        
        # New Reward Shaping Hyperparameters for Acrobot
        self.reward_shaping_params = {
            'gamma': self.gamma,
            'lambda_height': 1.0,
            'alpha_action_cost': 0.01,
            'xi_velocity_penalty': 0.05,
            'plan_bonus': 5.0,
            'athena_bonus': 2.0,  # New bonus from Athena rules
            'athena_penalty': -2.0 # New penalty from Athena rules
        }

        # Core Networks
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=state_dim, nhead=2, batch_first=True)
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
        
        self.meta_learner = None

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
        self.archive = {}
        self.faiss_index = faiss.IndexFlatL2(self.latent_dim)
        self.faiss_features = []
        self.fun_history = deque(maxlen=100)
        
        # Symbolic Planner and Knowledge Base
        self.planner = SymbolicPlanner()
        self.symbolic_knowledge_base = {
            'rules': []
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
        for training. This version is for un-guided action selection.
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

    def _select_guided_action(self, state, guidance):
        """
        Selects an action by biasing the policy towards the planner's guidance.
        """
        features, hidden = self.encode(state.unsqueeze(0))
        with torch.no_grad():
            probs, value = self.policy(features)
        
        biased_probs = probs + torch.tensor(guidance, device=probs.device).float() * 0.5
        biased_probs = torch.softmax(biased_probs, dim=-1)
        
        dist = Categorical(biased_probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=probs.device))
        
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
        dopamine = 0.01 + math.tanh(fun) + (1 if reward > 10 else 0) * 0.5
        self.fun_history.append(reward)
        return fun * dopamine
        
    def _get_potential(self, state):
        """
        Computes the potential function for Acrobot, based on height.
        state is a numpy array.
        """
        if state is None:
            return 0
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, _, _ = state
        return -cos_theta1 - cos_theta1 * cos_theta2 + sin_theta1 * sin_theta2

    def compute_shaped_reward(self, r_orig, current_state, next_state, action):
        """
        Computes a new, shaped reward based on the provided comprehensive suite.
        Uses a combination of potential-based and direct shaping, now
        including a bonus for following the symbolic plan and Athena's rules.
        """
        r_shaped = r_orig

        # 1. Height-Based Potential Shaping (ensures policy invariance)
        potential_s = self._get_potential(current_state)
        potential_s_prime = self._get_potential(next_state)
        r_potential = self.reward_shaping_params['gamma'] * potential_s_prime - potential_s
        r_shaped += r_potential * self.reward_shaping_params['lambda_height']

        # 2. Action Cost Penalty (a direct, non-potential shaping)
        r_action_cost = -self.reward_shaping_params['alpha_action_cost'] * abs(action - 1)
        r_shaped += r_action_cost

        # 3. Stability Penalty Near Goal (based on angular velocities at high height)
        if next_state[0] < -0.8:
            omega1_sq = next_state[4] ** 2
            omega2_sq = next_state[5] ** 2
            r_stability = -self.reward_shaping_params['xi_velocity_penalty'] * (omega1_sq + omega2_sq)
            r_shaped += r_stability

        # 4. Symbolic Plan Bonus
        if self.planner.check_sub_goal(next_state):
             r_shaped += self.reward_shaping_params['plan_bonus']
             self.planner.advance_sub_goal()
        
        # 5. Athena Symbolic Reward (New)
        r_shaped += self.apply_symbolic_reward(next_state, action)
        
        return r_shaped
        
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

    def get_policy_loss(self, experiences):
        """
        Calculates the PPO policy and value loss for a batch of experiences.
        Used by the MetaLearner for the inner loop.
        """
        states, actions, rewards, _, dones, log_probs, values = zip(*experiences)
        
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
        
        features, _ = self.encode(states)
        probs, values = self.policy(features)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values.squeeze(-1), returns)
        
        return policy_loss + 0.5 * value_loss

    def _imagination_rollout(self, start_state, rollout_length=15):
        """
        Generates a sequence of imagined states, actions, and rewards.
        This rollout is now also guided by the planner's sub-goals.
        """
        imagined_experiences = []
        current_state_np = start_state.numpy()
        
        self.planner.current_sub_goal_idx = 0 
        
        for _ in range(rollout_length):
            guidance = self.planner.provide_guidance(current_state_np)
            action, log_prob, value, _ = self._select_guided_action(torch.tensor(current_state_np).float(), guidance)

            imagined_next_state, imagined_reward = self.dreamer.predict(current_state_np, action)
            
            total_imagined_reward = self.compute_shaped_reward(imagined_reward, current_state_np, imagined_next_state, action)
            
            imagined_exp = Experience(current_state_np, action, total_imagined_reward, imagined_next_state, False, log_prob.item(), value.item())
            imagined_experiences.append(imagined_exp)

            current_state_np = imagined_next_state

        return imagined_experiences

    def _update_policy_from_imagination(self, imagined_experiences):
        """
        Updates the policy using the PPO loss calculated on imagined experiences.
        """
        if not imagined_experiences:
            return 0
        
        for _ in range(self.ppo_epochs):
            loss = self.get_policy_loss(imagined_experiences)
            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()

        return loss.item()

    def update_policy(self):
        """This function is now superseded by _update_policy_from_imagination"""
        return 0

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

        for _ in range(self.critic_iters):
            self.optimizer_critic.zero_grad()
            
            real_state = torch.tensor(good_exp.state).float()
            real_action = torch.tensor([good_exp.action]).float()
            real_reward = torch.tensor([good_exp.reward]).float()
            real_next_state = torch.tensor(good_exp.next_state).float()
            real = torch.cat([real_state, real_action, real_reward, real_next_state]).unsqueeze(0)
            
            generated = self.gan_rehab.generate(bad_exp).unsqueeze(0)

            d_real = self.gan_rehab.critic(real.detach())
            d_fake = self.gan_rehab.critic(generated.detach())
            
            loss_d = d_fake.mean() - d_real.mean()
            
            loss_d.backward()
            self.optimizer_critic.step()
            for p in self.gan_rehab.critic.parameters():
                p.data.clamp_(-0.01, 0.01)

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
            fake_exp = Experience(s.detach().numpy(), random.randint(0,2), 10, s.detach().numpy(), False, None, None)
            self.main_buffer.push(fake_exp, priority=2.0)

    def sync_buffers(self):
        """Synchronizes experiences from good/bad buffers to the main buffer."""
        for exp in self.good_buffer.buffer:
            self.main_buffer.push(exp, priority=2.0)
        for exp in self.bad_buffer.buffer:
            self.main_buffer.push(exp, priority=0.5)
        self.good_buffer.buffer.clear()
        self.bad_buffer.buffer.clear()
        
    def clone(self):
        """Creates a deep copy of the agent for the inner loop."""
        return copy.deepcopy(self)

    def athena_extract(self):
        """
        Analyzes recent experiences from the good and bad buffers to extract
        and store symbolic, causal rules for Acrobot.
        """
        if len(self.good_buffer) < 1 or len(self.bad_buffer) < 1:
            return
        
        # Extract experiences from the good buffer
        good_experiences = random.sample(self.good_buffer.buffer, min(10, len(self.good_buffer)))
        good_states = np.array([exp.state for exp in good_experiences])
        good_actions = np.array([exp.action for exp in good_experiences])
        good_next_states = np.array([exp.next_state for exp in good_experiences])

        # Extract experiences from the bad buffer
        bad_experiences = random.sample(self.bad_buffer.buffer, min(10, len(self.bad_buffer)))
        bad_states = np.array([exp.state for exp in bad_experiences])
        bad_actions = np.array([exp.action for exp in bad_experiences])
        bad_next_states = np.array([exp.next_state for exp in bad_experiences])

        # 1. Learn Rules for Positive Height Change (Desirable)
        # We look for a consistent action that leads to a height increase
        # The height is based on the goal: -cos(theta1) - cos(theta1 + theta2)
        def get_height(states):
            return -states[:, 0] - (states[:, 0] * states[:, 2] - states[:, 1] * states[:, 3])
        
        good_height_change = get_height(good_next_states) - get_height(good_states)
        
        for action in range(self.action_dim):
            action_indices = np.where(good_actions == action)[0]
            if len(action_indices) > 0:
                avg_change = np.mean(good_height_change[action_indices])
                if avg_change > 0.05: # Threshold for a significant positive change
                    rule = {
                        'if': {'action': action, 'effect': 'positive_height_change'},
                        'then': {'outcome': 'desirable', 'reward_bonus': self.reward_shaping_params['athena_bonus']}
                    }
                    if rule not in self.symbolic_knowledge_base['rules']:
                        self.symbolic_knowledge_base['rules'].append(rule)
        
        # 2. Learn Rules for High Angular Velocity (Undesirable)
        # We look for actions that lead to excessively high velocities
        high_velocity_threshold = 5.0
        
        for action in range(self.action_dim):
            action_indices = np.where(bad_actions == action)[0]
            if len(action_indices) > 0:
                avg_velocity_1 = np.mean(np.abs(bad_next_states[action_indices, 4]))
                avg_velocity_2 = np.mean(np.abs(bad_next_states[action_indices, 5]))
                if avg_velocity_1 > high_velocity_threshold or avg_velocity_2 > high_velocity_threshold:
                    rule = {
                        'if': {'action': action, 'effect': 'high_angular_velocity'},
                        'then': {'outcome': 'undesirable', 'reward_penalty': self.reward_shaping_params['athena_penalty']}
                    }
                    if rule not in self.symbolic_knowledge_base['rules']:
                        self.symbolic_knowledge_base['rules'].append(rule)

    def apply_symbolic_reward(self, current_state, current_action):
        """
        Uses the learned symbolic knowledge to provide an additional reward signal.
        """
        symbolic_reward = 0.0
        
        current_height = -current_state[0] - (current_state[0] * current_state[2] - current_state[1] * current_state[3])
        
        for rule in self.symbolic_knowledge_base['rules']:
            # Check for desirable rules
            if rule['if'].get('action') == current_action:
                if rule['if'].get('effect') == 'positive_height_change':
                    # We can't know the 'before' state, so we check if the current state
                    # is in a good configuration for this rule to be meaningful.
                    if current_height > -1.0: # Check that we are not at the lowest point
                        symbolic_reward += rule['then']['reward_bonus']
            
            # Check for undesirable rules
            if rule['if'].get('effect') == 'high_angular_velocity':
                if (np.abs(current_state[4]) > 5.0 or np.abs(current_state[5]) > 5.0):
                    symbolic_reward += rule['then']['reward_penalty']
                    
        return symbolic_reward
    
    def train_step(self, env):
        """
        Refactored training step with the Symbolic Planner.
        1. Collect a small batch of real-world experience, guided by the planner.
        2. Update the world model components.
        3. Perform policy updates using imagined rollouts, which are also guided.
        """
        
        real_experiences = []
        state, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        
        self.planner.current_sub_goal_idx = 0 
        
        while not done and not truncated and step_count < 100:
            
            guidance = self.planner.provide_guidance(state)
            action, log_prob, value, _ = self._select_guided_action(torch.tensor(state).float(), guidance)
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Now, we extract Athena's rules from the experiences
            self.athena_extract()

            # The shaped reward now includes Athena's symbolic bonus/penalty
            shaped_reward = self.compute_shaped_reward(reward, state, next_state, action)
            
            fun_score = self.compute_fun_score(shaped_reward, [shaped_reward], self.encode(torch.tensor(state).float().unsqueeze(0))[0])
            icm_reward = self.update_icm([(state, action, shaped_reward, next_state, done, None, None)])
            
            total_reward = shaped_reward + self.alpha_fun * fun_score + self.alpha_icm * icm_reward
            exp = Experience(state, action, total_reward, next_state, done, log_prob.item(), value.item())
            self.main_buffer.push(exp)
            
            real_experiences.append(exp)
            state = next_state
            step_count += 1
            self.steps += 1
            
        if len(self.main_buffer) > self.batch_size:
            batch, _, _ = self.main_buffer.sample(self.batch_size)
            self.dreamer.update(batch)
            for head_idx in range(3):
                self.update_q(batch, head_idx)
            self.rehab_experiences()
            self.dream_generate()

        if real_experiences:
            start_state_np = random.choice(real_experiences).state
            start_state_tensor = torch.tensor(start_state_np).float()
            
            imagined_experiences = self._imagination_rollout(start_state_tensor, rollout_length=50)
            policy_loss = self._update_policy_from_imagination(imagined_experiences)
            
            self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        
        return {'episode_reward': sum([exp.reward for exp in real_experiences]), 'policy_loss': policy_loss}

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
                'rules': []
            })
            self.faiss_features = state['faiss_features']
            
            self.faiss_index = faiss.IndexFlatL2(self.latent_dim)
            if self.faiss_features:
                faiss_features_np = np.array(self.faiss_features).astype('float32').reshape(-1, self.latent_dim)
                self.faiss_index.add(faiss_features_np)
