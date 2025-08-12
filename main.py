# main.py

import os
import gym
import torch
import numpy as np
from caosb_world_model.agents.world_model_agent import WorldModelBuilder
from caosb_world_model.agents.meta_learner import MetaLearner
from caosb_world_model.core.experience import Experience

# --- Hyperparameters ---
HYPERPARAMETERS = {
    'latent_dim': 128,
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 1e-4,
    'meta_lr': 1e-5,  # New: learning rate for meta-update
    'inner_lr': 1e-3, # New: learning rate for inner adaptation
    'num_inner_updates': 1, # New: number of gradient steps in inner loop
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

# --- Task Generation ---
def create_task_env(gravity_modifier=1.0, main_engine_power_modifier=1.0):
    """
    Creates a LunarLander environment with modified physical parameters.
    This defines a 'task' for our meta-learning.
    """
    env = gym.make("LunarLander-v2", new_step_api=True)
    env.gravity *= gravity_modifier
    env.main_engine_power *= main_engine_power_modifier
    return env

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
    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = WorldModelBuilder(state_dim, action_dim, HYPERPARAMETERS).to(device)
    
    # Initialize meta-learner
    meta_learner = MetaLearner(agent, meta_lr=HYPERPARAMETERS['meta_lr'])

    num_meta_epochs = 100
    tasks_per_batch = 5
    
    for meta_epoch in range(num_meta_epochs):
        print(f"--- Starting Meta-Epoch {meta_epoch + 1}/{num_meta_epochs} ---")
        
        task_batch = []
        # Step 1: Create a batch of diverse tasks
        for _ in range(tasks_per_batch):
            # Define a random task by varying gravity and engine power
            gravity_mod = np.random.uniform(0.8, 1.2)
            engine_mod = np.random.uniform(0.8, 1.2)
            task_env = create_task_env(gravity_mod, engine_mod)
            task_batch.append(task_env)
        
        # Step 2: Perform the meta-training step
        meta_learner.meta_train(task_batch)
        
        # Step 3: Evaluate the original agent's performance on a held-out task
        eval_env = create_task_env(gravity_modifier=1.0, main_engine_power_modifier=1.0)
        eval_reward = 0
        state, _ = eval_env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(device)
            action, _, _, _ = agent.select_action(state_tensor)
            next_state, reward, done, truncated, _ = eval_env.step(action)
            done = done or truncated
            eval_reward += reward
            state = next_state
        
        print(f"Evaluation Reward (standard task): {eval_reward}")

    # You would typically save the final meta-learned agent here
    agent.save_pics("final_meta_agent.pth")
    print("Meta-training complete. Agent saved.")

if __name__ == "__main__":
    main()
