# main.py

import gymnasium as gym
import torch
import pandas as pd
import os

# Import the core components
from caosb_world_model.environments.shaped_lunar_lander import ShapedLunarLander
from caosb_world_model.agents.world_model_agent import WorldModelBuilder

# Define Hyperparameters in a single dictionary
HYPERPARAMETERS = {
    # Environment
    'state_dim': 8,
    'action_dim': 4,
    
    # Core Agent
    'latent_dim': 64,
    'buffer_capacity': 1000000,
    'batch_size': 64,
    'gamma': 0.99,
    'tau': 0.005,
    'lr': 0.001,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.995,
    'alpha_fun': 0.1,
    'alpha_icm': 0.1,
    
    # PPO
    'ppo_epochs': 4,
    'ppo_clip': 0.2,
    'ppo_lambda': 0.95,
}

def main():
    """
    Main training function for the CAOSB-World Model.
    """
    # 1. Setup Environment and Agent
    print("Setting up the environment and agent...")
    env = ShapedLunarLander(gym.make("LunarLander-v2"))
    agent = WorldModelBuilder(HYPERPARAMETERS['state_dim'], HYPERPARAMETERS['action_dim'], HYPERPARAMETERS)
    agent.athena_extract(env)
    
    # 2. Setup Logging and Telemetry
    print("Initializing logging...")
    log_df = pd.DataFrame(columns=['episode', 'reward'])
    telemetry = []

    # 3. Main Training Loop
    print("Starting training...")
    for episode in range(1000):
        # Perform a single training step (one full episode)
        metrics = agent.train_step(env)
        
        # Log episode results
        log_df = pd.concat([log_df, pd.DataFrame({'episode': [episode], 'reward': [metrics['episode_reward']]})], ignore_index=True)
        log_df.to_csv('training_logs.csv', index=False)
        
        # Log telemetry data (e.g., losses)
        telemetry.append({'episode': episode, 'q_loss': 0, 'policy_loss': metrics['policy_loss']})
        pd.DataFrame(telemetry).to_csv('telemetry.csv', index=False)
        
        print(f"Episode {episode}: Reward {metrics['episode_reward']:.2f}, Policy Loss {metrics['policy_loss']:.4f}")
        
        # 4. Checkpointing and Archiving
        if episode > 0 and episode % 100 == 0:
            checkpoint_path = f'checkpoint_{episode}.pth'
            print(f"Saving checkpoint to {checkpoint_path}")
            agent.save_pics(checkpoint_path)
            # Archiving the model state for later analysis
            agent.archive[f'model_{episode}'] = agent.state_dict()
        
        # 5. Curriculum Learning
        if metrics['episode_reward'] > 100 and env.stage < 3:
            env.stage += 1
            print(f"Advancing to curriculum stage {env.stage}")

    env.close()
    print("Training complete.")

if __name__ == '__main__':
    # Ensure the directory structure exists
    os.makedirs('caosb_world_model/agents', exist_ok=True)
    os.makedirs('caosb_world_model/core', exist_ok=True)
    os.makedirs('caosb_world_model/modules', exist_ok=True)
    os.makedirs('caosb_world_model/environments', exist_ok=True)
    
    # Create the __init__.py files
    with open('caosb_world_model/__init__.py', 'w') as f: pass
    with open('caosb_world_model/agents/__init__.py', 'w') as f: pass
    with open('caosb_world_model/core/__init__.py', 'w') as f: pass
    with open('caosb_world_model/modules/__init__.py', 'w') as f: pass
    with open('caosb_world_model/environments/__init__.py', 'w') as f: pass
    
    main()
