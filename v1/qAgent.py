import random
from Agent import Agent

__author__ = 'philippe'


class QAgent(Agent):
    def treat_new_data(self, new_data):
        # Add the difference to the data vector
        self.data.append(int(0.08 * (new_data - self.old_data)))

        # Delete the oldest data point if the vector is too big
        if len(self.data) > self.data_to_consider:
            self.data.pop(0)
        self.old_data = new_data

    def build_state(self, agent_is_holding):
        return str(agent_is_holding), str(self.data)

    def update(self, action, reward, new_agent_is_holding):
            self.new_state = self.build_state(new_agent_is_holding)
            max_act, max_val = self.max_q(self.new_state)
            self.inc_q(self.old_state, action, reward + self.gamma * max_val)

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

    def get_best_action(self, agent_is_holding):
        self.old_state = self.build_state(agent_is_holding)

        # Get the action that maximizes the Q function
        max_act, max_val = self.max_q(self.old_state)
        return max_act