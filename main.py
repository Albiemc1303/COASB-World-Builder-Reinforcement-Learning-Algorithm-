# main.py

import os
import gymnasium as gym
import torch
import numpy as np
from world_model_agent import WorldModelBuilder
from meta_learner import MetaLearner
from experience import Experience

# --- Hyperparameters ---
HYPERPARAMETERS = {
    'latent_dim': 128,
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 1e-4,
    'meta_lr': 1e-5,
    'inner_lr': 1e-3,
    'num_inner_updates': 1,
    'alpha_fun': 0.1,
    'alpha_icm': 0.1,
    'ppo_epochs': 4,
    'ppo_clip': 0.2,
    'ppo_lambda': 0.95,
    'buffer_capacity': 100000,
    'batch_size': 64,
    'eps_start': 0.9,
    'eps_end': 0.05,
    'eps_decay': 1000,
    'critic_iters': 5,
    'gan_lr': 1e-4
}

# --- Task Generation for Acrobot ---
def create_task_env(start_range=0.1):
    """
    Creates an Acrobot environment with a modified starting state range.
    This defines a 'task' for meta-learning.
    """
    env = gym.make("Acrobot-v1")
    
    # We will wrap the environment to modify the initial state
    class AcrobotTaskWrapper(gym.Wrapper):
        def __init__(self, env, start_range):
            super().__init__(env)
            self.start_range = start_range

        def reset(self, **kwargs):
            # First call the parent reset to properly initialize the environment
            obs, info = self.env.reset(**kwargs)
            
            # For Acrobot, the state has 4 elements: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
            # But the internal state might be different. Let's modify the observation instead.
            # Apply the start_range scaling to the observation
            scaled_obs = obs * self.start_range
            
            return scaled_obs, info

    return AcrobotTaskWrapper(env, start_range)


def get_task_experiences(env, agent, num_episodes=5):
    """
    Runs episodes in a given environment to collect a batch of experiences.
    """
    experiences = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        hidden = None
        while not done:
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            action, log_prob, value, hidden = agent.get_ppo_action(state_tensor, hidden=hidden)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            exp = Experience(state, action, reward, next_state, done, log_prob, value)
            experiences.append(exp)
            state = next_state
            
    return experiences


# --- Main Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize main agent
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]  # 6
    action_dim = env.action_space.n           # 3
    agent = WorldModelBuilder(state_dim, action_dim, HYPERPARAMETERS).to(device)
    
    # Initialize meta-learner
    meta_learner = MetaLearner(agent, meta_lr=HYPERPARAMETERS['meta_lr'])

    num_meta_epochs = 100
    tasks_per_batch = 5
    
    for meta_epoch in range(num_meta_epochs):
        print(f"--- Starting Meta-Epoch {meta_epoch + 1}/{num_meta_epochs} ---")
        
        task_batch = []
        # Step 1: Create a batch of diverse tasks by varying the starting state range
        for _ in range(tasks_per_batch):
            start_range_mod = np.random.uniform(0.01, 0.3)
            task_env = create_task_env(start_range_mod)
            task_batch.append(task_env)
        
        # Step 2: Perform the meta-training step
        meta_learner.meta_train(task_batch)
        
        # Step 3: Evaluate the original agent's performance on a standard task
        eval_env = gym.make("Acrobot-v1")
        eval_reward = 0
        state, _ = eval_env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            action, _ = agent.select_action(state_tensor)
            next_state, reward, done, truncated, _ = eval_env.step(action)
            done = done or truncated
            eval_reward += reward
            state = next_state
        
        print(f"Evaluation Reward (standard task): {eval_reward}")

    agent.save_pics("final_acrobot_meta_agent.pth")
    print("Meta-training complete. Agent saved.")

if __name__ == "__main__":
    main()
