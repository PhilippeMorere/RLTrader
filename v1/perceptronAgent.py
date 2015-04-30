from Agent import Agent
__author__ = 'philippe'


class PerceptronAgent(Agent):
    def __init__(self, actions):
        Agent.__init__(self, actions)
        self.weights = [[1.0] * (self.data_to_consider + 1)] * len(self.actions)
        print self.weights

    def update(self, action, reward, new_agent_is_holding):
        # BAD: TO DO
        next_max_val, next_max_act = self.get_best_action(new_agent_is_holding)
        max_val, max_act = self.getValue(state)
        increment_q = reward + self.gamma * next_max_val

        for i in range(0, len(self.featureQ)):
            #self.w[action][i] *= 1 - self.alpha
            self.w[action][i] += self.alpha * (increment_q - max_val) * self.featureQ[i](state)

        # self.new_state = self.build_state(new_agent_is_holding)
        # max_act, max_val = self.max_q(self.new_state)
        # self.inc_q(self.old_state, action, reward + self.gamma * max_val)

    def get_best_action(self, agent_is_holding):
        inputs = self.data[:]
        if agent_is_holding:
            inputs.append(1.0)
        else:
            inputs.append(0.0)

        max_val = None
        max_act = None
        for act in range(1, len(self.actions)):
            sum = 0
            for i in range(1, len(inputs)):
                sum += self.weights[act][i] * inputs[i]
            if max_val is None or max_val < sum:
                max_val = sum
                max_act = self.actions[act]
        return max_act

    def treat_new_data(self, new_data):
        # Add the difference to the data vector
        self.data.append(new_data - self.old_data)
        #print self.data

        # Delete the oldest data point if the vector is too big
        if len(self.data) > self.data_to_consider:
            self.data.pop(0)
        self.old_data = new_data
