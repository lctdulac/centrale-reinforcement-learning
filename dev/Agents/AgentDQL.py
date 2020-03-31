from tensorflow import keras
import numpy as np
from random import random, randint, sample
import matplotlib.pyplot as plt


class AgentDQL:

    """
    In this class we create a Deep Q-learning Agent, 
    running in a stochastic MDP environment.
    It uses Tensorflow Backend for Deep Learning.
    ________________________________________________________________
    Parameters :
    - gamma : float [0,1]
    The discount factor
    - lr : float [0,1]
    The learning rate
    - eps : float [0,1] 
    The exploration factor
    - episodes : int
    Number of episodes to run
    - batch_size : int
    Number of samples to take as a batch for training the Q network
    - train_every : int
    Train the Q network every () steps
    - update_every : int
    Update the target Network every () steps
    - env : MDP_environment
    Environment to run the Agent in

    ________________________________________________________________
    Attributes :
    - s : int
    The current state
    - durations : list[int]
    List of each episodes' duration
    - episode_memory_buffer : list[list[int]]
    List of each episodes' states history list
    - episode_memory_buffer_len : int
    Size of the path memory buffer for the display of histories
    - memory : list[tuple(state, action, reward, new_state)]
    Memory buffer
    - memory_size : int (default=2000)
    Size of the memory buffer for the tuples (state, action, reward, new_state)
    - network : Keras model
    Main Q network
    - target_network : Keras model
    Target Q network

    """

    def __init__(self, gamma, lr, eps, episodes, batch_size, train_every, update_every, env):

        # Parameters
        self.episodes = episodes
        self.gamma = gamma
        self.lr = lr
        self.eps = eps
        self.batch_size = batch_size
        self.train_every = train_every
        self.update_every = update_every
        self.env = env

        # Attributes
        self.s = 0  # Initialization : state 0
        self.durations = []
        self.episode_memory_buffer = []
        self.episode_memory_buffer_len = 3

        # Deep Q specific attributes
        self.memory = []
        self.memory_size = 2000
        self.network = self.create_network()
        self.target_network = self.create_network()


    def create_network(self):

        """ 
        Instantiates the Q-network 
        as a two hidden layers dense network.
        """

        model = keras.Sequential([
            keras.layers.Dense(24, input_dim = self.env.getNbStates(), activation="relu"),
            keras.layers.Dense(12, activation = "relu"),
            keras.layers.Dense(self.env.getNbActions())])
        model.compile(loss = "mean_squared_error", optimizer = 'adam')

        return model


    def state_to_input(self, state):

        """
        One-hot encoding function for the states.
        """

        input = [0 for i in range(self.env.getNbStates())]
        input[state] = 1
        return [input]


    def replay(self, i):

        """
        Uses memory replay to update the target and train the model with a batch.
        """

        # Make sure we have enough memory for training

        if len(self.memory) < self.batch_size:
            return

        # Training

        print("==>", "(", i, ")" ," Training network...")

        memory_samples = sample(self.memory, self.batch_size)

        for m in memory_samples:
            state, action, reward, new_state = m

            # update target

            target = self.target_network.predict(self.state_to_input(state))
            if new_state == -1:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * max(self.target_network.predict(self.state_to_input(new_state))[0])

            # train network

            self.network.fit(
                np.array(self.state_to_input(state)), 
                target, 
                epochs=1, 
                verbose=0)


    def replace_network(self, i):

        """ Updates Target Network. """

        print("==>", "(", i, ")" ," Updating target network...")

        weights = self.network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i]

        self.target_network.set_weights(target_weights)


    def getNextAction(self):

        rand = random()

        # exploitation
        if rand > self.eps:  
            a = np.argmax(self.network.predict(self.state_to_input(self.s))[0])
        # exploration
        else:  
            a = randint(0, 3)
        return a


    def learning(self):

        """
        Main training loop.
        """

        print("=== USING DEEP Q TRAINING ===")

        # Environment parameters

        dims = self.env.getDims()
        n_actions = self.env.getNbActions()
        n_states = self.env.getNbStates()
        s_final = self.env.get_final_state()

        # Training

        self.rewards_hist = []
        step_counter = 0

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

            while self.s != -1:

                # Sample action, get next state

                a = self.getNextAction()

                next_state, reward = self.env.next_step(self.s, a)

                episode_hist.append(next_state)

                episode_reward += reward

                # Write in memory

                if next_state == s_final:
                    next_state = -1

                if len(self.memory) < self.memory_size:
                    self.memory.append([self.s, a, reward, next_state])
                else:
                    self.memory = self.memory[1:] + [[self.s, a, reward, next_state]]

                # Next state update

                self.s = next_state

                # Train the Q-Network every S steps

                if step_counter % self.train_every == 0:
                    self.replay(step_counter)
                
                if step_counter % self.update_every == 0:
                    self.replace_network(step_counter)

                step_counter += 1

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
        print("=== DEEP-Q TRAINING FINISHED ===")


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
