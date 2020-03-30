import numpy as np
from random import random, randint
import matplotlib.pyplot as plt

from Agents.AgentQL import AgentQL


class AgentSARSA(AgentQL):

    """ 
    In this class we create a SARSA Agent, 
    inheritating from the AgentQ Q-learning Agent.
    The only difference is in the training loop. 

    See the documentation of AgentQ for more details.
    """

    def learning(self):

        print("=== USING SARSA TRAINING ===")

        # Environment parameters

        dims = self.env.getDims()
        n_actions = self.env.getNbActions()
        n_states = self.env.getNbStates()
        s_final = self.env.get_final_state()

        # Training

        self.rewards_hist = []

        for i in range(self.episodes):

            # Epsilon Decay

            if i == self.episodes//4:
                print("dividing by two")
                self.eps /= 2

            if i == self.episodes//2:
                print("dividing by two")
                self.eps /= 2

            if i == 3*self.episodes//4:
                print("dividing by two")
                self.eps /= 2

            if i == 9*self.episodes//10:
                print("removing random exploration")
                self.eps = 0

            # Episode Initialization

            print("=> épisode %i/%i..." % (i+1, self.episodes))

            episode_reward = 0
            episode_hist = [0]

            # State and action intialization

            self.s = 0  # S
            a = self.getNextAction(self.s)  # A

            while self.s != s_final:

                # Get next state

                step = self.env.next_step(self.s, a)

                next_state = step[0] # S'

                episode_hist.append(next_state)

                reward = step[1]

                episode_reward += reward

                next_action = self.getNextAction(next_state)  # A'

                # Compute target

                if next_state == s_final:
                    target = self.env.get_final_reward()
                else:
                    target = reward + self.gamma * self.Q[next_state, next_action]

                # Q-matrix update

                self.Q[self.s, a] = (1-self.lr) * \
                    self.Q[self.s, a] + self.lr * target

                # Next state update

                self.s = next_state
                a = next_action

            # End of the episode

            print("===> duration: ", len(episode_hist))
            print("===> reward: ", episode_reward)

            # History update

            self.durations.append(len(episode_hist))
            self.rewards_hist.append(episode_reward)

            if len(self.episode_memory_buffer) == self.episode_memory_buffer_len:
                self.episode_memory_buffer = self.episode_memory_buffer[1:] + [
                    episode_hist]
            else:
                self.episode_memory_buffer += [episode_hist]

        # End of training
        print("=== SARSA TRAINING FINISHED ===")