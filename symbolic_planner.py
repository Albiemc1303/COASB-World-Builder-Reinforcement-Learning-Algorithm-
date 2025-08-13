# agents/symbolic_planner.py

import numpy as np

class SymbolicPlanner:
    def __init__(self):
        # A simple, hand-coded symbolic representation of the Acrobot state
        # In a more advanced system, this would be learned by the agent.
        self.symbolic_states = {
            'low_energy_down': lambda s: s[0] > 0.5 and s[4] < 1.0,
            'building_energy': lambda s: s[0] < 0.5 and abs(s[4]) > 1.0,
            'swinging_up': lambda s: s[0] < -0.5 and s[4] > 0,
            'near_goal': lambda s: -s[0] - s[2] > 0.5
        }
        
        # A hard-coded symbolic plan to achieve the goal
        # This is a sequence of sub-goals
        self.plan = [
            'build_energy',
            'swinging_up',
            'near_goal'
        ]
        self.current_sub_goal_idx = 0

    def get_current_sub_goal(self):
        """Returns the current symbolic sub-goal."""
        if self.current_sub_goal_idx < len(self.plan):
            return self.plan[self.current_sub_goal_idx]
        return 'goal_achieved'

    def advance_sub_goal(self):
        """Moves to the next sub-goal in the plan."""
        self.current_sub_goal_idx += 1
        if self.current_sub_goal_idx >= len(self.plan):
            return True # Plan is complete
        return False # Plan is not complete
        
    def check_sub_goal(self, state):
        """
        Checks if the current state satisfies the conditions for the current sub-goal.
        """
        sub_goal_name = self.get_current_sub_goal()
        if sub_goal_name == 'goal_achieved':
            return True
        
        check_func = self.symbolic_states.get(sub_goal_name)
        if check_func:
            return check_func(state)
        return False

    def provide_guidance(self, state):
        """
        Provides guidance to the policy based on the current sub-goal.
        Returns a suggested action or a "focus" vector.
        """
        sub_goal_name = self.get_current_sub_goal()
        
        if sub_goal_name == 'build_energy':
            # Suggest a pumping action. This is a heuristic.
            return np.array([1, 0, 0]) # e.g., a bias towards positive or negative torque
        elif sub_goal_name == 'swinging_up':
            # Focus on positive torque when the arm is swinging upwards
            if state[4] > 0: # Check angular velocity
                return np.array([0, 0, 1]) # bias towards action 2 (+1 torque)
            else:
                return np.array([1, 0, 0]) # bias towards action 0 (-1 torque)
        elif sub_goal_name == 'near_goal':
            # Focus on a balanced action (0 torque)
            return np.array([0, 1, 0]) # bias towards action 1 (0 torque)
        
        return np.array([0.33, 0.33, 0.33]) # default, no bias

