from tensorflow import keras
import numpy as np
from random import random, randint, sample, seed
import matplotlib.pyplot as plt

# set the seed so we generate the same grid
seed(42)


class AgentDQ:

    def __init__(self, gamma, lr, eps, episodes, dims, env):
        # paramètres
        self.gamma = gamma
        self.lr = lr
        # (si eps=0, greedy policy) compris entre 0 et 1. si rand < eps : exploration
        self.eps = eps

        # matrices
        self.env = env
        self.dims = dims
        self.n_actions = self.env.T.shape[0]
        self.n_states = self.dims[0] * self.dims[1]

        # agent
        self.s = 0  # initialisation à l'état 0
        self.episodes = episodes
        self.durations = []

        self.memory = []
        self.memory_size = 2000

        self.episode_memory_buffer = []
        self.episode_memory_buffer_len = 3

        self.network = self.create_network()
        self.target_network = self.create_network()

    def create_network(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.n_states, activation="relu"),
            keras.layers.Dense(12, activation="relu"),
            keras.layers.Dense(self.n_actions)])
        model.compile(loss="mean_squared_error", optimizer='adam')
        return model

    def state_to_input(self, state):
        input = [0 for i in range(self.n_states)]
        input[state] = 1
        return [input]

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        print("training network")
        memory_samples = sample(self.memory, batch_size)
        for mem in memory_samples:
            state, action, reward, new_state = mem
            target = self.target_network.predict(self.state_to_input(state))
            if new_state == -1:
                target[0][action] = action
            else:
                target[0][action] = reward + self.gamma * max(self.target_network.predict(self.state_to_input(new_state))[0])
            self.network.fit(np.array(self.state_to_input(state)), target, epochs=1, verbose=0)

    def replace_network(self):
        weights = self.network.get_weights()
        target_weights = self.target_network.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_network.set_weights(target_weights)

    def getNextAction(self):
        rand = random()
        if rand > self.eps:  # exploitation
            a = np.argmax(self.network.predict(self.state_to_input(self.s))[0])
        else:  # exploration
            a = randint(0, 3)
        return a

    def DQ_Learning(self):

        s_final = self.env.final_state()
        self.rewards_hist = []

        step_counter = 0

        # entraînement

        for i in range(self.episodes):

            # epsilon decay

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

            # Initialisation de l'épisode

            print("- épisode %i/%i..." % (i, self.episodes))

            episode_reward = 0
            episode_hist = [0]

            # Initialisation
            self.s = 0

            while self.s != -1:

                # sampling de l'action et obtention de l'état suivant

                a = self.getNextAction()

                next_state, reward = self.env.next_step(self.s, a)

                episode_hist.append(next_state)

                episode_reward += reward

                if next_state == s_final:
                    next_state = -1

                if len(self.memory) < self.memory_size:
                    self.memory.append([self.s, a, reward, next_state])
                else:
                    self.memory = self.memory[1:] + [[self.s, a, reward, next_state]]

                # passage à l'état suivant
                self.s = next_state

                if step_counter % 20 == 0:
                    self.replay()
                    self.replace_network()

                step_counter += 1

            # fin de l'épisode

            print("--> Durée de l'épisode : ", len(episode_hist),
                  ", reward :", episode_reward)

            # historiques

            self.durations.append(len(episode_hist))
            self.rewards_hist.append(episode_reward)

            print(episode_hist)
            if len(self.episode_memory_buffer) == self.episode_memory_buffer_len:
                self.episode_memory_buffer = self.episode_memory_buffer[1:] + [
                    episode_hist]
            else:
                self.episode_memory_buffer += [episode_hist]

        # fin de l'entraînement
        # print(self.Q)

    def plot_history(self):

        plt.plot(range(self.episodes), self.durations)
        plt.figure(2)
        plt.plot(range(self.episodes), self.rewards_hist)
        plt.show()
