"""
Defines all environments used in the experiments. Includes a custom 4x4 grid
written from scratch and wrappers for Gymnasium environments. All environments
follow the same interface so the training code works the same for all of them.
"""

import numpy as np
import gymnasium as gym


class CustomGridWorld:
    """
    A simple deterministic 4x4 grid environment built from scratch.

    Grid layout (states 0 to 15, row major):
        0   1   2   3
        4  [X]  6   7      X = hole at state 5, reward -1, episode ends
        8   9  10  11
       12  13  14  [G]     G = goal at state 15, reward +1, episode ends

    Start is state 0. Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    Step penalty is -0.01 to push the agent toward shorter paths.

    This environment mimics the Gymnasium interface so it works with
    value_iteration(), train_agent(), and run_experiment() without any changes.
    """

    def __init__(self):
        self.nrow = 4
        self.ncol = 4
        self.n_states = 16
        self.n_actions = 4
        self.start_state = 0
        self.goal_state = 15
        self.hole_states = {5}
        self.state = self.start_state

        self.observation_space = type('Space', (), {'n': self.n_states})()
        self.action_space = type('Space', (), {
            'n': self.n_actions,
            'sample': lambda: np.random.randint(4)
        })()

        self.P = {}
        for s in range(self.n_states):
            self.P[s] = {}
            for a in range(self.n_actions):
                self.P[s][a] = self._get_transition(s, a)

    def _get_transition(self, state, action):
        """
        Returns the transition tuple for a given state and action.
        Terminal states loop back to themselves with reward 0.
        Boundary moves keep the agent in the same cell.
        Returns a list with one tuple: (probability, next_state, reward, done).
        """
        if state == self.goal_state or state in self.hole_states:
            return [(1.0, state, 0.0, True)]

        row, col = state // self.ncol, state % self.ncol

        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.ncol - 1, col + 1)
        elif action == 2:
            row = min(self.nrow - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)

        next_state = row * self.ncol + col

        if next_state == self.goal_state:
            return [(1.0, next_state, +1.0, True)]
        elif next_state in self.hole_states:
            return [(1.0, next_state, -1.0, True)]
        else:
            return [(1.0, next_state, -0.01, False)]

    def reset(self, seed=None):
        """
        Resets the agent to state 0. Accepts a seed for compatibility
        with Gymnasium but transitions are deterministic so it has no effect.
        Returns (initial_state, empty_info_dict).
        """
        if seed is not None:
            np.random.seed(seed)
        self.state = self.start_state
        return self.state, {}

    def step(self, action):
        """
        Executes the action and returns (next_state, reward, terminated, truncated, info).
        Looks up the precomputed transition from self.P.
        """
        transitions = self.P[self.state][action]
        prob, next_state, reward, done = transitions[0]
        self.state = next_state
        return next_state, reward, done, False, {}

    @property
    def unwrapped(self):
        """Returns self so value_iteration() can access env.unwrapped.P."""
        return self


ENV_CONFIGS = {
    'CustomGrid-4x4': {
        'desc': 'Custom 4x4 GridWorld (deterministic)',
        'type': 'custom',
    },
    'FrozenLake-4x4-det': {
        'desc': 'FrozenLake 4x4 (deterministic)',
        'type': 'gym',
        'id': 'FrozenLake-v1',
        'kwargs': {'map_name': '4x4', 'is_slippery': False},
    },
    'FrozenLake-4x4-slip': {
        'desc': 'FrozenLake 4x4 (slippery/stochastic)',
        'type': 'gym',
        'id': 'FrozenLake-v1',
        'kwargs': {'map_name': '4x4', 'is_slippery': True},
    },
    'FrozenLake-8x8-det': {
        'desc': 'FrozenLake 8x8 (deterministic)',
        'type': 'gym',
        'id': 'FrozenLake-v1',
        'kwargs': {'map_name': '8x8', 'is_slippery': False},
    },
    'FrozenLake-8x8-slip': {
        'desc': 'FrozenLake 8x8 (slippery/stochastic)',
        'type': 'gym',
        'id': 'FrozenLake-v1',
        'kwargs': {'map_name': '8x8', 'is_slippery': True},
    },
    'CliffWalking': {
        'desc': 'CliffWalking (risk environment)',
        'type': 'gym',
        'id': 'CliffWalking-v0',
        'kwargs': {},
    },
    'Taxi': {
        'desc': 'Taxi-v3 (500 states, scaling test)',
        'type': 'gym',
        'id': 'Taxi-v3',
        'kwargs': {},
    },
}


def make_env(name, seed=None):
    """
    Creates and returns an environment by name along with its config dict.

    Accepts any key from ENV_CONFIGS. Raises ValueError if the name is unknown.
    Returns (env, config).
    """
    if name not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown environment: '{name}'. "
            f"Choose from: {list(ENV_CONFIGS.keys())}"
        )
    config = ENV_CONFIGS[name]
    if config['type'] == 'custom':
        env = CustomGridWorld()
    else:
        env = gym.make(config['id'], **config['kwargs'])
    return env, config
