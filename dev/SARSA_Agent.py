import numpy as np
from scipy import stats
from random import random, randint, seed
import matplotlib.pyplot as plt

# set the seed so we generate the same grid
seed(42)


class AgentSARSA:

    # cas stochastique avec T et tirage aléatoire
    # cas déterministe à faire?

    def __init__(self, gamma, lr, eps, episodes, dims, T, R):

        # paramètres
        self.gamma = gamma
        self.lr = lr
        # (si eps=0, greedy policy) compris entre 0 et 1. si rand < eps : exploration
        self.eps = eps

        # matrices
        self.T = T
        self.R = R
        self.dims = dims
        self.n_actions = self.T.shape[0]
        self.n_states = self.dims[0] * self.dims[1]
        # Q : lignes : états, colonnes : actions
        self.Q = np.zeros((self.n_states, self.n_actions))

        # agent
        self.s = 0  # initialisation à l'état 0
        self.episodes = episodes
        self.durations = []

        self.episode_memory_buffer = []
        self.episode_memory_buffer_len = 3

    def getNextAction(self, s):
        Q_s = self.Q[s, :]  # shape (4)
        rand = random()
        if rand > self.eps:  # exploitation
            a = np.argmax(Q_s)
        else:  # exploration
            a = randint(0, 3)
        return a

    def getNextState(self, a):

        # cas stochastique

        distribution_prochain_etat = self.T[a, self.s, :].tolist()

        probs, etats = [], []
        for j in range(len(distribution_prochain_etat)):
            if distribution_prochain_etat[j] != 0:
                probs.append(distribution_prochain_etat[j])
                etats.append(j)

        custm = stats.rv_discrete(name="custm", values=(etats, probs))
        prochain_etat = custm.rvs(size=1)

        return prochain_etat[0]

    def SARSA_Learning(self):

        # initialisation à 0 mieux que random

        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         self.Q[s, a] = random()

        s_final = np.argmax(self.R)
        self.rewards_hist = []

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

            self.s = 0  # S
            a = self.getNextAction(self.s)  # A

            while self.s != s_final:

                next_state = self.getNextState(a)  # S'

                episode_hist.append(next_state)

                reward = self.R[next_state // self.dims[1],
                                next_state % self.dims[1]]  # R

                episode_reward += reward

                next_action = self.getNextAction(next_state)  # A'

                # calcul de la cible
                if next_state == s_final:
                    target = self.R[s_final // self.dims[1],
                                    s_final % self.dims[1]]
                else:
                    target = reward + self.gamma * \
                        self.Q[next_state, next_action]

                # update de la Q-table
                self.Q[self.s, a] = (1-self.lr) * \
                    self.Q[self.s, a] + self.lr * target

                # passage à l'état suivant
                self.s = next_state
                a = next_action

            # fin de l'épisode

            print("--> Durée de l'épisode : ", len(episode_hist),
                  ", reward :", episode_reward)

            # historiques

            self.durations.append(len(episode_hist))
            self.rewards_hist.append(episode_reward)

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
