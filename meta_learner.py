# caosb_world_model/agents/meta_learner.py

import torch
import torch.optim as optim
import copy
from collections import deque
import random
from experience import Experience

class MetaLearner:
    """
    Implements a Model-Agnostic Meta-Learning (MAML) training loop for the WorldModelBuilder.
    
    The goal is to train the agent to quickly adapt to new tasks, such as
    different environment dynamics, by optimizing its initial parameters.
    """
    def __init__(self, agent, meta_lr=1e-5):
        """
        Args:
            agent (WorldModelBuilder): The main agent to be meta-trained.
            meta_lr (float): The learning rate for the meta-update (outer loop).
        """
        self.agent = agent
        self.meta_optimizer = optim.Adam(self.agent.parameters(), lr=meta_lr)

    def meta_train(self, task_batch, num_inner_updates=1, inner_lr=1e-3):
        """
        Performs a full meta-training step on a batch of tasks.
        
        Args:
            task_batch (list): A list of tasks (e.g., environments with different parameters).
            num_inner_updates (int): Number of gradient steps for the inner loop.
            inner_lr (float): The learning rate for the inner loop.
        """
        meta_update_grads = []

        for task in task_batch:
            # Step 1: Create a temporary clone for fast adaptation
            task_agent = self.agent.clone()
            task_optimizer = optim.Adam(task_agent.parameters(), lr=inner_lr)

            # Step 2: Inner loop (Adapt to the task)
            task_experiences = self.get_task_experiences(task, task_agent)
            for _ in range(num_inner_updates):
                task_loss = self.calculate_task_loss(task_agent, task_experiences)
                task_optimizer.zero_grad()
                task_loss.backward()
                task_optimizer.step()

            # Step 3: Evaluate adapted agent and get gradients for meta-update
            # We'll use a validation set for this, or a different batch of experiences from the same task
            val_experiences = self.get_task_experiences(task, task_agent)
            val_loss = self.calculate_task_loss(task_agent, val_experiences)
            val_loss.backward()

            # Collect the gradients from the adapted agent's parameters
            meta_update_grads.append({name: p.grad for name, p in task_agent.named_parameters()})

        # Step 4: Outer loop (Meta-Update the original agent)
        self.meta_optimizer.zero_grad()
        # Sum gradients from all tasks and apply them to the original agent
        for name, p in self.agent.named_parameters():
            if name in meta_update_grads[0] and meta_update_grads[0][name] is not None:
                # Only sum gradients that are not None
                valid_grads = [g[name] for g in meta_update_grads if g[name] is not None]
                if valid_grads:
                    p.grad = sum(valid_grads) / len(valid_grads)
        self.meta_optimizer.step()

    def get_task_experiences(self, task, agent=None, num_episodes=5):
        """
        Runs episodes in a given task environment to collect a batch of experiences.
        
        Args:
            task: The task environment to collect experiences from
            agent: The agent to use for collecting experiences (defaults to self.agent)
            num_episodes: Number of episodes to run
            
        Returns:
            List of Experience objects
        """
        if agent is None:
            agent = self.agent
            
        experiences = []
        
        for _ in range(num_episodes):
            state, _ = task.reset()
            done = False
            hidden = None
            while not done:
                state_tensor = torch.tensor(state).float().unsqueeze(0)
                action, log_prob, value, hidden = agent.get_ppo_action(state_tensor, hidden=hidden)
                next_state, reward, done, truncated, info = task.step(action)
                done = done or truncated
                
                exp = Experience(state, action, reward, next_state, done, log_prob, value)
                experiences.append(exp)
                state = next_state
                
        return experiences

    def calculate_task_loss(self, agent, experiences):
        """
        Placeholder for calculating a loss (e.g., policy or value loss)
        on a specific set of experiences.
        """
        # Example: a simple PPO loss on the batch
        # This will be handled by the agent's methods
        return agent.get_policy_loss(experiences)

