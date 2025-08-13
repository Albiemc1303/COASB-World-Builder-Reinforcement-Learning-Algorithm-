# caosb_world_model/core/buffers.py

import numpy as np
import random
from collections import deque
from experience import Experience

class ExperienceReplayBuffer:
    """
    Experience replay buffer with optional Prioritized Experience Replay (PER).
    
    PER uses a priority queue to sample more important experiences more frequently,
    which can accelerate learning. The importance is determined by the TD-error.
    """
    def __init__(self, capacity, prioritized=False):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.prioritized = prioritized
        if prioritized:
            self.priorities = np.zeros((capacity,), dtype=np.float32)
            self.alpha = 0.6  # PER alpha: determines how much prioritization is used
            self.beta = 0.4   # PER beta: determines how much importance sampling is used
            self.epsilon = 1e-5 # Small constant to prevent zero priority

    def push(self, experience: Experience, priority=1.0):
        """Adds an experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        if self.prioritized:
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of experiences.
        Returns samples, indices, and importance-sampling weights if using PER.
        """
        if not self.prioritized:
            if len(self.buffer) < batch_size:
                return [], [], []
            samples = random.sample(self.buffer, batch_size)
            return samples, None, None
        
        if len(self.buffer) < batch_size:
            return [], [], []

        # Calculate probabilities based on priorities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalize weights
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Updates the priorities for a batch of experiences."""
        for idx, prio in zip(indices, priorities):
            if idx < self.capacity:
                self.priorities[idx] = prio + self.epsilon

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)
