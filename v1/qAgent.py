import random

__author__ = 'philippe'


class QAgent:
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

    def treat_new_data(self, new_data):
        # Add the difference to the data vector
        self.data.append(int(0.08 * (new_data - self.old_data)))

        # Delete the oldest data point if the vector is too big
        if len(self.data) > self.data_to_consider:
            self.data.pop(0)
        self.old_data = new_data

    def build_state(self, agent_is_holding):
        return str(agent_is_holding), str(self.data)

    def reward(self, action, reward, new_agent_is_holding, new_data):
        self.balance += reward

        # Create the new state
        self.treat_new_data(new_data)
        self.new_state = self.build_state(new_agent_is_holding)

        if not self.old_state:
            pass
        else:
            max_act, max_val = self.max_q(self.new_state)
            self.inc_q(self.old_state, action, reward + self.gamma * max_val)

        self.old_state = self.new_state

    def get_action(self, agent_is_holding, data):
        if len(self.data) < self.data_to_consider:
            return self.actions[2]  # Hold if not enough data

        self.old_state = self.build_state(agent_is_holding)

        # Get the action that maximizes the Q function
        max_act, max_val = self.max_q(self.old_state)

        # Explore ?
        if random.random <= self.epsilon:
            return self.actions[int(random.random()*3)]
        else:
            return max_act

    def max_q(self, state):
        # Never seen this state before, random action
        if not (state in self.Q):
            return self.actions[int(random.random()*3)], 0.1

        val = None
        act = None
        for a, q in self.Q[state].items():
            if val is None or (q > val):
                val = q
                act = a
        return act, val

    def inc_q(self, state, action, increment):
        # Add the state to Q if it doesn't exist
        if not (state in self.Q):
            temp = {}
            for action in self.actions:
                temp[action] = 0.0
            self.Q[state] = temp

        # Update the Q table
        self.Q[state][action] *= 1 - self.alpha
        self.Q[state][action] += self.alpha * increment

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