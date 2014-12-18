import math
import random
import qAgent
import matplotlib.pyplot as plt

__author__ = 'philippe'


class DataGenerator:
    def __init__(self):
        self.sinus_offset = 1.2
        self.sinus_period = 0.1
        self.time = 0

    def generate_sinus_data(self):
        sin1 = 10 * math.sin(self.sinus_offset + self.sinus_period * self.time)
        sin2 = -7 * math.sin(self.sinus_offset + 8 * self.sinus_period * self.time)
        sin3 = 5 * math.sin(self.sinus_offset + 1 + 20 * self.sinus_period * self.time)
        sin4 = -1 * math.sin(self.sinus_offset + 2.1 + 6 * self.sinus_period * self.time)
        point = sin1 + sin2 + sin3 + sin4 + random.random() * 10
        return point

    def generate_increasing_data(self):
        return self.time * 0.1

    def increase_time(self):
        self.time += 1

    def is_first_pass(self):
        return self.time == 0


class World:
    actions = ['buy', 'hold', 'sell']

    def __init__(self):
        self.data_generator = DataGenerator()
        self.agent = qAgent.QAgent(self.actions)
        self.agent_is_holding = False
        self.gap = 2.0  # %
        self.not_trading_fee = 0.1  # %
        self.number_training = 1000000
        self.number_test = 400
        self.all_actions = []
        self.data_generated = []

    def main(self):
        reward = 0
        old_data = 0
        action = self.actions[1]
        generation = 0
        is_test = False
        while generation < (self.number_training + self.number_test):
            # Generate new data
            new_data = self.data_generator.generate_sinus_data()
            if is_test:
                self.data_generated.append(new_data)

            # For the old round
            if not self.data_generator.is_first_pass():
                # Compute reward
                reward = self.compute_reward(action, old_data, new_data)
                if is_test:
                    self.all_actions.append(self.agent_is_holding)

                # Reward agent
                self.agent.reward(action, reward, self.agent_is_holding, new_data)

            # Get action
            action = self.agent.get_action(self.agent_is_holding, new_data)

            old_data = new_data
            self.data_generator.increase_time()
            #time.sleep(0.01)
            generation += 1
            if generation % 10000 == 0 and generation > 0:
                if generation <= self.number_training:
                    print "Training generation", str(generation), "/", self.number_training
                else:
                    print "Test generation", str(generation - self.number_training), "/", self.number_test
                self.agent.display_info()
            if generation == self.number_training:
                is_test = True
                self.agent.disable_training()

        self.agent.display_info()
        #self.agent.print_best_states(15)
        self.plot_data()

    def plot_data(self):
        plt.plot(range(len(self.data_generated)), self.data_generated, 'b-')
        plt.plot(range(len(self.all_actions)), self.all_actions, 'r-')
        plt.show()

    def compute_reward(self, action, old_data, new_data):
        earnings = new_data - old_data

        # Decrease the agent's earnings when it doesn't trade, force it to trade
        if not self.agent_is_holding:
            earnings = - self.not_trading_fee * math.fabs(new_data) / 100.0

        # Update the agent state: holding or not
        if action == self.actions[0]:
            # Apply a 2% gap whenever the agent takes a position
            if not self.agent_is_holding:
                earnings -= new_data * self.gap / 100.0
            self.agent_is_holding = True
        elif action == self.actions[2]:
            self.agent_is_holding = False

        return earnings

world = World()
world.main()