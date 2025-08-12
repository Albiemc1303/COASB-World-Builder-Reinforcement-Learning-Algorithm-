# caosb_world_model/core/experience.py

from collections import namedtuple

# Define a named tuple for storing an experience
# It includes all the information needed for training different components (DQN, PPO)
Experience = namedtuple('Experience', [
    'state', 
    'action', 
    'reward', 
    'next_state', 
    'done', 
    'log_prob', 
    'value'
])

