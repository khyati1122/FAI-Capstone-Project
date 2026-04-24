"""
From scratch implementations of Q learning, SARSA, Expected SARSA,
Double Q learning, and value iteration. All algorithms store Q values
in a 2D numpy array of shape (n_states, n_actions) and use epsilon
greedy exploration.
"""

import numpy as np


class BaseAgent:
    """
    Shared base class for all four TD control algorithms.

    Handles Q table setup, epsilon greedy action selection, and epsilon decay.
    Subclasses only need to implement the update() method with their own learning rule.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))
        self.training_rewards = []
        self.q_history = []

    def select_action(self, state):
        """
        Epsilon greedy action selection.
        With probability epsilon picks a random action, otherwise picks
        the action with the highest Q value in the current state.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self):
        """
        Reduces epsilon by the decay factor at the end of each episode.
        Epsilon will never go below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """
        Returns the greedy action for every state based on the current Q table.
        Output shape is (n_states,).
        """
        return np.argmax(self.q_table, axis=1)

    def get_q_table(self):
        """Returns a copy of the current Q table so external code cannot modify it."""
        return self.q_table.copy()

    def reset_q_table(self):
        """Resets the Q table and all training logs back to their initial state."""
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.training_rewards = []
        self.q_history = []


class QLearning(BaseAgent):
    """
    Q learning (Watkins and Dayan 1992).

    Off policy TD control. The update target always uses the maximum Q value
    of the next state, regardless of which action the agent actually takes next.
    This means it learns the optimal policy even while exploring randomly.

    Update rule: Q(s,a) += alpha * (r + gamma * max Q(s',a') - Q(s,a))
    """
    name = "Q-learning"

    def update(self, state, action, reward, next_state, done):
        """
        One step Q learning update.

        If done is True the target is just r since there is no next state.
        Otherwise the target is r + gamma * max Q value at next state.
        Returns the TD error (how wrong the current estimate was).
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


class SARSA(BaseAgent):
    """
    SARSA (Rummery and Niranjan 1994).

    On policy TD control. The name comes from the five values used each update:
    State, Action, Reward, next State, next Action.

    Unlike Q learning, SARSA uses the Q value of the action the agent WILL
    actually take next, not the theoretical maximum. This makes it more
    conservative near dangerous states like cliffs because it accounts for
    the chance of accidentally taking a bad action during exploration.

    Update rule: Q(s,a) += alpha * (r + gamma * Q(s', a') - Q(s,a))
    where a' is the actual next action chosen by the current policy.
    """
    name = "SARSA"

    def update(self, state, action, reward, next_state, done, next_action=None):
        """
        One step SARSA update.

        Requires next_action because the target uses Q(next_state, next_action)
        not the max. The training loop must select next_action before calling this.
        Returns the TD error.
        """
        if done:
            td_target = reward
        else:
            if next_action is None:
                next_action = self.select_action(next_state)
            td_target = reward + self.gamma * self.q_table[next_state, next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


class ExpectedSARSA(BaseAgent):
    """
    Expected SARSA (Van Seijen et al 2009).

    Instead of using one sampled next action (SARSA) or the max (Q learning),
    this takes the weighted average Q value across all possible next actions
    under the current epsilon greedy policy. This reduces variance compared
    to SARSA without introducing the overestimation bias of Q learning.

    Under epsilon greedy policy:
        Expected Q(s) = (1 - epsilon) * max Q(s,a) + epsilon/n_actions * sum Q(s,a)
    """
    name = "Expected SARSA"

    def _expected_q(self, state):
        """
        Computes the expected Q value at a state under the current epsilon greedy policy.
        Returns a float representing the weighted average Q value.
        """
        q_values = self.q_table[state]
        best_action = np.argmax(q_values)
        policy_probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        policy_probs[best_action] += (1.0 - self.epsilon)
        return np.dot(policy_probs, q_values)

    def update(self, state, action, reward, next_state, done):
        """
        One step Expected SARSA update using the expected Q value at next state.
        Returns the TD error.
        """
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self._expected_q(next_state)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return td_error


class DoubleQLearning(BaseAgent):
    """
    Double Q learning (Van Hasselt 2010).

    Regular Q learning overestimates Q values because using max over noisy
    estimates is always higher than the true max (Jensen's inequality).
    Double Q learning fixes this by keeping two separate Q tables. One table
    selects the best action, the other evaluates how good that action is.
    Since the two tables are updated independently, the bias cancels out.

    Action selection uses the sum of both tables.
    On each update step, Q1 or Q2 is updated with 50% probability, using
    the other table for evaluation.
    """
    name = "Double Q-learning"

    def __init__(self, *args, **kwargs):
        """Sets up two independent Q tables in addition to the parent Q table."""
        super().__init__(*args, **kwargs)
        self.q_table1 = np.zeros((self.n_states, self.n_actions))
        self.q_table2 = np.zeros((self.n_states, self.n_actions))
        self.q_table = (self.q_table1 + self.q_table2) / 2.0

    def select_action(self, state):
        """
        Epsilon greedy selection using the combined estimate from both Q tables.
        Returns the chosen action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        combined = self.q_table1[state] + self.q_table2[state]
        return int(np.argmax(combined))

    def update(self, state, action, reward, next_state, done):
        """
        One step Double Q learning update.

        With 50% probability updates Q1 using Q2 for evaluation,
        otherwise updates Q2 using Q1 for evaluation.
        The shared q_table is kept as the average of Q1 and Q2.
        Returns the TD error.
        """
        if np.random.random() < 0.5:
            if done:
                td_target = reward
            else:
                best_a = np.argmax(self.q_table1[next_state])
                td_target = reward + self.gamma * self.q_table2[next_state, best_a]
            td_error = td_target - self.q_table1[state, action]
            self.q_table1[state, action] += self.alpha * td_error
        else:
            if done:
                td_target = reward
            else:
                best_a = np.argmax(self.q_table2[next_state])
                td_target = reward + self.gamma * self.q_table1[next_state, best_a]
            td_error = td_target - self.q_table2[state, action]
            self.q_table2[state, action] += self.alpha * td_error

        self.q_table = (self.q_table1 + self.q_table2) / 2.0
        return td_error

    def get_mean_overestimation(self, q_star):
        """
        Computes how much the agent overestimates Q values relative to Q*.
        Returns the mean difference between max Q(s) and max Q*(s) across all states.
        Positive means overestimation, negative means underestimation.
        """
        max_q = np.max(self.q_table, axis=1)
        max_q_star = np.max(q_star, axis=1)
        return np.mean(max_q - max_q_star)

    def reset_q_table(self):
        """Resets both Q tables and all logs. Overrides parent to clear Q1 and Q2."""
        super().reset_q_table()
        self.q_table1 = np.zeros((self.n_states, self.n_actions))
        self.q_table2 = np.zeros((self.n_states, self.n_actions))


def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=10000):
    """
    Computes the exact optimal Q values using dynamic programming.

    This is not a learning algorithm. It reads the full transition table P[s][a]
    directly from the environment and iteratively applies the Bellman equation
    until values converge. We use the result as ground truth to measure how
    accurate each learning algorithm's Q table is.

    Requires env.P or env.unwrapped.P to be accessible.

    Returns Q_star (n_states x n_actions), V_star (n_states,), and the
    optimal policy (n_states,).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    if hasattr(env, 'P'):
        P = env.P
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'P'):
        P = env.unwrapped.P
    else:
        raise ValueError(
            "Environment does not expose transition dynamics (P). "
            "Value iteration requires a model of the environment."
        )

    V = np.zeros(n_states)

    for iteration in range(max_iterations):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            q_sa = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    q_sa[a] += prob * (reward + gamma * (0 if done else V[next_state]))
            V_new[s] = np.max(q_sa)
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new

    Q_star = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            for prob, next_state, reward, done in P[s][a]:
                Q_star[s, a] += prob * (reward + gamma * (0 if done else V[next_state]))

    policy = np.argmax(Q_star, axis=1)
    return Q_star, V, policy
