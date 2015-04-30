import random
__author__ = 'philippe'


class Agent:
    def __init__(self, actions):
        self.actions = actions
        self.data = []
        self.data_to_consider = 6
        self.old_data = 0
        self.Q = dict()
        self.epsilon = 0.05
        self.alpha = 0.2
        self.gamma = 0.8
        self.old_state = None
        self.new_state = None
        self.balance = 0

    def get_action(self, agent_is_holding):
        if len(self.data) < self.data_to_consider:
            return self.actions[2]  # Hold if not enough data

        # Explore ?
        if random.random <= self.epsilon:
            return self.actions[int(random.random()*3)]
        else:
            # Best action
            return self.get_best_action(agent_is_holding)

    def get_best_action(self, agent_is_holding):
        return self.actions[2]  # Do nothing: Hold

    def treat_new_data(self, new_data):
        # Do nothing
        pass

    def update(self, action, reward, new_agent_is_holding):
        # Do nothing
        pass

    def reward(self, action, reward, new_agent_is_holding, new_data):
        self.balance += reward

        # Create the new state
        self.treat_new_data(new_data)

        if not self.old_state:
            pass
        else:
            self.update(action, reward, new_agent_is_holding)

        self.old_state = self.new_state

    def disable_training(self):
        print "######### End of training"
        self.balance = 0
        self.epsilon = 0
        self.alpha = 0

    def display_info(self):
        print "Profit:", str(int(self.balance*100)/100.0)
        print "Number of states:", len(self.Q)

    def print_best_states(self, mini):
        nb = 0
        for state in self.Q:
            for a, q in self.Q[state].items():
                if q > mini:
                    print state, a
                    nb += 1
        print nb

