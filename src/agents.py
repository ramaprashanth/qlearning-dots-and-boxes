from players import Player
import numpy as np
from numpy import random
from environment import Game
from copy import copy
from dqn import DQN


class QAgent(Player):
    """
    QAgent inherits base player and uses Q-Learning
    """
    def __init__(self, name, alpha=1e-4, epsilon=0.05, gamma=0.6, learning=True, dqn=False):
        super().__init__()
        self.name = name
        self.learning = learning
        self.dqn = False

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.QTable = dict()
        self.DQN = None

        self.default_action = None
        self.action_dict = dict()

        self.log_file = None

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, environment):
        self._environment = environment
        self.default_action = tuple([0]*len(self._environment.action_list))
        self.action_dict = dict()
        for index,action in enumerate(self._environment.action_list):
            self.action_dict[action] = index

    def act(self):
        """
        Chooses an action according to the QTable and performs an update if possible
        """
        state = self._environment.state
        assert state is not None, 'Invalid State Passed for Agent Name: {}'.format(self.name)
        # feature_vector -> the current state of the environment (nxnxd numpy array)
        feature_vector = self.generate_input_vector(state)
        action = self.choose_action(feature_vector)
        self.last_state = feature_vector.copy()
        self.last_action = action
        self._environment.step(action)
        return action

    def observe(self, state, reward):
        """
        Observe state-reward pair
        """
        if self.learning:
            have_next_turn = int(self._environment.current_player == self)

            if self._environment.state is not None:
                next_feature_vector = self.generate_input_vector(state)
            else:
                next_feature_vector = None

            self.update(self.last_state, self.last_action, next_feature_vector, reward, have_next_turn)

    def generate_input_vector(self, state):
        """Generates an input vector based on an environment state."""
        r, c, d = state.shape
        input_vector = np.resize(state, [1, r, c, d])

        return input_vector

    def choose_action(self, feature_vector):
        """
        Selects a possible action with max Q value
        """
        if self.dqn:
            # DQN
            q_values = self.DQN.predict(feature_vector)[0] # Get all Q-Values for the current state
            max_valid_q = q_values[self._environment.valid_actions].max() # Get the highest Q value that corresponds to a valid action
            best_actions = np.where(q_values == max_valid_q)[0] # Select all actions that have this Q-Value
            chosen_action = random.choice([action for action in best_actions if action in self._environment.valid_actions]) # Choose randomly among valid actions
        else:
            # QTable
            feature_vector = tuple(feature_vector.reshape(-1))
            q_values = self.QTable.setdefault(feature_vector, self.default_action)
            possible_q_values = [q_values[self.action_dict[action]] for action in self._environment.valid_actions]
            max_index = possible_q_values.index(max(possible_q_values))
            max_q_action = self._environment.valid_actions[max_index]
            chosen_action = 0
            if random.random() >= self.epsilon and self.learning:
                chosen_action = max_q_action
            else:
                poss_actions = copy(self._environment.valid_actions)
                chosen_action = poss_actions.pop(max_index)
                if len(poss_actions) == 1 :
                    chosen_action = poss_actions[-1]
                elif len(poss_actions) > 1:
                    chosen_action = poss_actions[random.randint(0,len(poss_actions)-1)]
            return chosen_action

    def update(self, current_state, last_action, next_state, reward, have_next_turn):
        """
        Updates the Qtable or DQN
        """
        if self.dqn:
            self.DQN.record_state((current_state, last_action, next_state, reward, have_next_turn))
            self.DQN.train()
        else:
            if next_state is not None:
                current_state = tuple(current_state.reshape(-1))
                next_state = tuple(next_state.reshape(-1))
                q_current_all = list(self.QTable.setdefault(current_state, self.default_action))
                current_action_index = self.action_dict[last_action]
                q_current = q_current_all[current_action_index]
                q_current_max  = max(self.QTable.setdefault(next_state, self.default_action))
                q_new = q_current + self.alpha * (reward + self.gamma * q_current_max - q_current)
                q_current_all[current_action_index] = q_new
                self.QTable[current_state] = tuple(q_current_all)

    def initialize_network(self, output='tanh'):
        """
        Create the DQN
        """
        assert self._environment is not None, 'Cannot initialize a network without environment'
        input_shape = self._environment.state.shape
        outputs = len(self._environment.action_list)
        self.DQN = DQN(input_shape, outputs, alpha=self.alpha, gamma=self.gamma, output=output)
