from numpy import random

class Player:
    """
    Base Player for a game.
    """

    def __init__(self,name=None):
        """Initialize player variables"""
        self._environment = None
        self.name = name

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self,environment):
        self._environment = environment

    def __str__(self):
        return self.name

    def act(self):
        """Takes a random action in the environment."""
        if self._environment == None:
            raise ValueError("Must add an environment in order to act")

        #Take an action randomly
        action = self.choose_action()
        self._environment.step(action)
        return action

    def observe(self,last_state,reward):
        """Observe the environment and do nothing"""
        pass

    def choose_action(self):
        """Choose an action randomly in the environment."""
        action = random.choice(self._environment.valid_actions)

        return action


class RandomPlayer(Player):
    """
    RandomPlayer that checks puts 4th line to close box and
    avoids 3rd line to lead to a box.
    """
    def __init__(self, name, level=1):
        super().__init__(name=name)
        self.level = level

    def choose_action(self):
        fourth_line_actions = []
        third_line_actions = []

        for action in self._environment.valid_actions:
            for line_count in self.analyze_action(action):
                if line_count == 4:
                    fourth_line_actions.append(action)
                if line_count == 3:
                    third_line_actions.append(action)
        #We want to avoid putting the third line on any box, because that means the opponent can score
        safe_actions = [a for a in self._environment.valid_actions if a not in third_line_actions]

        if len(fourth_line_actions) != 0:
            action = random.choice(fourth_line_actions)
        elif len(safe_actions) != 0 and self.level == 2:
            action = random.choice(safe_actions)
        else:
            action = random.choice(self._environment.valid_actions)

        return action


    def analyze_action(self,action):
        """
        Checks if the action will place line to complete one box
        and place the 3rd line on another, returns [4,3]
        """
        affected_cells = self._environment.convert_to_state(action)
        resulting_line_count = []
        for cell in affected_cells:
            row,column,side = cell
            current_lines = sum(self._environment.state[row,column])
            resulting_line_count.append(current_lines + 1)

        return resulting_line_count
