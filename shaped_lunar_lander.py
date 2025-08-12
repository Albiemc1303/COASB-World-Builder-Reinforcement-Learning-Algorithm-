# caosb_world_model/environments/shaped_lunar_lander.py

import gymnasium as gym
import numpy as np

class ShapedLunarLander(gym.Wrapper):
    """
    A Gymnasium wrapper for the LunarLander-v2 environment with curriculum-based
    reward shaping.
    
    The shaping changes based on the training stage, guiding the agent
    towards a successful landing step-by-step.
    """
    def __init__(self, env, stage=1):
        super().__init__(env)
        self.stage = stage

    def step(self, action):
        """
        Performs a step in the environment and applies reward shaping.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Unpack the state observation
        x, y, vx, vy, theta, omega, leg1, leg2 = obs
        shaped = 0
        
        if self.stage == 1:
            # Stage 1: Encourage gentle vertical speed and being above a certain height
            shaped += 5 if abs(vy) < 0.1 and y > 0.5 else -1
        elif self.stage == 2:
            # Stage 2: Encourage a controlled descent with low horizontal velocity
            shaped += 3 if vy < 0 and abs(vx) < 0.2 else -1
        elif self.stage == 3:
            # Stage 3: Encourage a stable, two-leg landing
            shaped += 10 if leg1 > 0 and leg2 > 0 and abs(vy) < 0.3 else -5
        
        # Penalty for tilting
        shaped -= abs(theta) * 0.5
        
        return obs, reward + shaped, done, truncated, info
