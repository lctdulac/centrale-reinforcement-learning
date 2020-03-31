import numpy as np
from random import random, randint
import matplotlib.pyplot as plt


class AgentQL:

    """
    In this class we create a Q-learning Agent, 
    running in a stochastic MDP environment.
    ________________________________________________________________
    Parameters :
    - gamma : float [0,1]
    The discount factor
    - lr : float [0,1]
    The learning rate
    - eps : float [0,1] 
    The exploration factor
    - episodes : int (default=2)
    Number of episodes to run
    - env : MDP_environment
    Environment to run the Agent in

    ________________________________________________________________
    Attributes :
    - Q : numpy array [n_states, n_actions]
    The Q table
    - s : int
    The current state
    - durations : list[int]
    List of each episodes' duration
    - episode_memory_buffer : list[list[int]]
    List of each episodes' states history list
    - episode_memory_buffer_len : int
    Size of the memory buffer for the display of histories 
    """

    def __init__(self, gamma, lr, eps, episodes, env):

        # Parameters
        self.episodes = episodes
        self.gamma = gamma
        self.lr = lr
        self.eps = eps
        self.env = env

        # Attributes
        self.Q = np.zeros((self.env.getNbStates(), self.env.getNbActions()))
        self.s = 0  # Initialization : state 0
        self.durations = []
        self.episode_memory_buffer = []
        self.episode_memory_buffer_len = 3


    def getNextAction(self, state):

        Q_s = self.Q[state, :]
        rand = random()

        # exploitation
        if rand > self.eps:  
            a = np.argmax(Q_s)
        # exploration
        else:  
            a = randint(0, 3)
        return a


    def learning(self):

        print("=== USING CLASSIC Q TRAINING ===")

        # Environment parameters

        dims = self.env.getDims()
        n_actions = self.env.getNbActions()
        n_states = self.env.getNbStates()
        s_final = self.env.get_final_state()

        # Training

        self.rewards_hist = []

        for i in range(self.episodes):

            # Epsilon decay

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

            # State initialization

            self.s = 0

            while self.s != s_final:

                # Sample action, get next state

                a = self.getNextAction(self.s)

                next_state, reward = self.env.next_step(self.s, a)

                episode_hist.append(next_state)

                episode_reward += reward

                # Compute target

                if next_state == s_final:
                    target = self.env.get_final_reward()
                else:
                    target = reward + self.gamma*np.max(self.Q[next_state, :])

                # Q-matrix update

                self.Q[self.s, a] = (1-self.lr) * \
                    self.Q[self.s, a] + self.lr * target

                # Next state update

                self.s = next_state

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
        print("=== Q TRAINING FINISHED ===")


    def plot_history(self, grid=False):

        # Plot duration and reward history

        plt.subplot(2, 1, 1)
        plt.plot(range(self.episodes), self.durations)
        plt.xlabel('episode')
        plt.ylabel('duration (ms)')
        plt.title('Episode duration evolution with training')
        plt.grid(grid)

        plt.subplot(2, 1, 2)
        plt.plot(range(self.episodes), self.rewards_hist)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.title('Episode reward evolution with training')
        plt.grid(grid)

        plt.tight_layout()
        plt.show()
        