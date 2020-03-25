import numpy as np
from scipy import stats
from random import random, randint, seed
import matplotlib.pyplot as plt

# set the seed so we generate the same grid
seed(42)


class AgentQ:

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
        self.history = [self.s]
        self.episodes = episodes
        self.durations = []

    def getNextAction(self):
        Q_s = self.Q[self.s, :]  # shape (4)
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

        self.history.append([self.s])

        return prochain_etat[0]

    def Q_Learning(self):

        # # initialisation

        # for s in range(self.n_states):
        #     for a in range(self.n_actions):
        #         self.Q[s, a] = random()

        # entraînement

        s_final = np.argmax(self.R)

        self.rewards_hist = []

        for i in range(self.episodes):

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

            print("- épisode %i/%i..." % (i, self.episodes))
            self.s = 0

            eps_reward = 0

            while self.s != s_final:

                # sampling de l'action et obtention de l'état suivant
                a = self.getNextAction()
                # print("action", a)
                next_state = self.getNextState(a)
                # print("next state", next_state)

                eps_reward += self.R[next_state // self.dims[1], next_state % self.dims[1]]

                # calcul de la cible
                if next_state == s_final:
                    target = self.R[s_final // self.dims[1],
                                    s_final % self.dims[1]]
                else:
                    target = self.R[next_state // self.dims[1],
                                    next_state % self.dims[1]] + self.gamma*np.max(self.Q[next_state, :])

                # update de la Q-table
                self.Q[self.s, a] = (1-self.lr) * \
                    self.Q[self.s, a] + self.lr * target

                # passage à l'état suivant
                self.s = next_state

            print("--> trajet : ", len(self.history), "rew :", eps_reward)
            self.durations.append(len(self.history))
            self.rewards_hist.append(eps_reward)
            self.history = [0]

        # print(self.Q)

    def plot_history(self):

        plt.plot(range(self.episodes), self.durations)
        plt.figure(2)
        plt.plot(range(self.episodes), self.rewards_hist)
        plt.show()
