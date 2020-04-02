import numpy as np
from random import random, randint, seed
from scipy import stats

# set the seed so we generate the same grid
seed(42)


class MDP_environment():
    """
    In this class we create a markov decision process (MDP) environment to run Reinforcement Learning experience
    ________________________________________________________________
    Parameters :
    - n_lin : int
    The number of line in the grid
    - n_col : int
    The number of column in the grid
    - p_obs : float [0,1] (default=0.15)
    The proportion of obstacles in the grid : for each state we draw a random number between 0 and 1
    and if it is <p_obs we consider it as an obstacle
    - n_traps : int (default=2)
    Number of traps we want in our map
    - p_coins : float [0,1] (default=0.1)
    Proportion of coins in the grid : for each state we draw a random number between 0 and 1
    and if it is <p_coins we consider it as a coin
    ________________________________________________________________
    Attributes :
    - actions : dict
    Dictionnary containing the actions that the agent can do. Every key is the name of the action,
    and the value is a list of probability to go to any adjacent state in this order : up, right,
    down, left
    - grid : numpy array
    Cointain the states of the MDP over which the agent can act
    - obstacles : list of tuple
    List cointaining the coordinate of the obstacles in the grid
    - treasure : tuple
    Coordiante of the tresure
    - traps : list of tuples
    List cointaining the coordinate of the traps in the grid
    - coins : list of tuples
    List cointaining the coordinate of the coins in the grid
    - T : numpy array
    3D Matrix of transition of the MDP. You need to call get_transition_matrix to compute this matrix
    - R : numpy array
    Reward matrix stocking the reward associatedto each state
    """

    def __init__(self, n_lin, n_col, p_obs=0.15, n_traps=2, p_coins=0.1):
        self.n_lin = n_lin
        self.n_col = n_col
        self.n_states = n_lin * n_col
        self.p_obs = p_obs
        self.n_traps = n_traps
        self.p_coins = p_coins
        self.make_grid(p_obs, n_traps, p_coins)
        self.actions = {'up':  [0.8, 0.1, 0., 0.1],
                        'right': [0.1, 0.8, 0.1, 0.],
                        'down': [0., 0.1, 0.8, 0.1],
                        'left': [0.1, 0., 0.1, 0.8]}
        self.set_rewards_matrix()
        self.set_transition_matrix()

    def getDims(self):
        return [self.n_lin, self.n_col]

    def getNbActions(self):
        return len(self.actions)

    def getNbStates(self):
        return self.getDims()[0]*self.getDims()[1]

    def make_grid(self, p_obs, n_traps, p_coins):
        self.grid = np.arange(self.n_states).reshape(self.n_lin, self.n_col)
        if p_obs != 0.0:
            self.set_obstables(p_obs)
        self.set_treasure()
        self.set_traps(n_traps)
        self.set_coins(p_coins)

    def set_obstables(self, p_obs):
        self.obstacles = []
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if i != 0 and j != 0:
                    p = random()
                    if p < p_obs:
                        self.grid[i, j] = -1
                        self.obstacles.append((i, j))

    def set_treasure(self):
        self.treasure = 0
        while self.treasure == 0:
            i, j = randint(self.n_lin-2, self.n_lin -
                           1), randint(self.n_col//2, self.n_col - 1)
            if (i, j) not in self.obstacles:
                self.treasure = (i, j)

    def set_traps(self, n_traps):
        self.traps = []
        while n_traps > 0:
            i, j = randint(0, self.n_lin - 1), randint(0, self.n_col - 1)
            if (i, j) not in self.obstacles and (i, j) != self.treasure:
                self.traps.append((i, j))
                n_traps -= 1

    def set_coins(self, p_coins):
        self.coins = []
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if (i, j) not in self.obstacles and (i, j) not in self.traps and (i, j) != self.treasure:
                    p = random()
                    if p < p_coins:
                        self.coins.append((i, j))

    def get_transition_matrix(self):
        return self.T

    def get_reward_matrix(self):
        return self.R

    def set_transition_matrix(self):
        # we first create an adjacent matrix to ease the construction of
        # the transition matrix
        adj = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            js = []
            if (i%self.n_col != 0):
                js.append(i-1)
            if (i%self.n_col != self.n_col-1):
                js.append(i+1)
            if (i // self.n_col != 0):
                js.append(i - self.n_col)
            if (i // self.n_col != self.n_lin - 1):
                js.append(i + self.n_col)
            for j in js:
                if self.grid[j//self.n_col, j % self.n_col] != -1:
                    adj[i, j] = 1
        # we can now contruct the transition matrix T
        T = []
        rotation_vect = [-self.n_col, 1, self.n_col, -1]
        for action in self.actions:
            Ta = np.zeros((self.n_states, self.n_states))
            prob_vector = self.actions[action]
            for i in range(self.n_states):
                if np.sum(adj[i, :]) > 1.0:
                    for k in range(len(prob_vector)):
                        j = i+rotation_vect[k]
                        if 0 <= j < self.n_states:
                            Ta[i, j] = adj[i, j]*prob_vector[k]
                elif np.sum(adj[i, :]) == 1.0:
                    Ta[i, :] = adj[i, :]
                else:
                    Ta[i, :] = np.zeros((len(Ta[i, :])))
                    Ta[i, i] = 1
                Lsum = np.sum(Ta[i, :])
                if Lsum != 1:
                    Ta[i, :] *= 1/Lsum
            T.append(Ta)
        self.T = np.array(T)

    def set_rewards_matrix(self):
        R = np.zeros((self.n_lin, self.n_col))
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if (i, j) in self.obstacles:
                    reward = 0
                elif (i, j) in self.traps:
                    reward = -1
                elif (i, j) == self.treasure:
                    reward = 1
                elif (i, j) in self.coins:
                    reward = 0.05
                else:
                    reward = -0.1
                R[i, j] = reward
        self.R = R

    def next_step(self, state, a):

        # Sample state from transition distribution (stochastic event)
        # Returns the next state and the step reward

        next_state_distrib = self.T[a, state, :].tolist()

        probs, states = [], []
        for j in range(len(next_state_distrib)):
            if next_state_distrib[j] != 0:
                probs.append(next_state_distrib[j])
                states.append(j)

        custm = stats.rv_discrete(name="custm", values=(states, probs))
        next_state = custm.rvs(size=1)

        return next_state[0], self.R[next_state[0] // self.n_col, next_state[0] % self.n_col]

    def to_coords(self, state):
        return state // self.n_col % self.n_lin, state % self.n_col

    def get_final_state(self):
        return np.argmax(self.R)

    def get_final_reward(self):
        return np.max(self.R)

    def print_grid_infos(self):
        print('Grid : \n', self.grid)
        print('\nTreasure coordinate : \n', self.treasure)
        print('\nObstacles coordinate : \n', self.obstacles)
        print('\nTraps coordinate : \n', self.traps)
        print('\nCoins coordinate : \n', self.coins)


class MDP_environment_simulated(MDP_environment):

    def __init__(self, n_lin, n_col):
        super().__init__(n_lin, n_col)
        self.coin_value = 0.5
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if 0 < self.R[i][j] < self.get_final_reward():
                    self.R[i][j] = self.coin_value

    def set_transition_matrix(self):
        # no more transition matrix
        pass

    def is_final(self, state):
        return np.argmax(state[:, :, 0]) == np.argmax(state[:, :, 1])

    def regenerate(self):
        self.make_grid(self.p_obs, self.n_traps, self.p_coins)
        self.set_rewards_matrix()
        self.coin_value = 0.5
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if 0 < self.R[i][j] < self.get_final_reward():
                    self.R[i][j] = self.coin_value

    def check_coord(self, coords):
        i, j = coords
        if min(i, j) < 0:
            return False
        if i >= self.n_lin:
            return False
        if j >= self.n_col:
            return False
        if self.R[i, j] == 0:
            return False
        return True

    def to_coords(self, state):
        return super().to_coords(np.argmax(state[:, :, 0]))

    def to_state(self, coords):
        player = np.zeros(np.shape(self.R))
        player[coords[0], coords[1]] = 1
        nstate = np.zeros((self.n_lin, self.n_col, 2))
        nstate[:, :, 0] = player
        nstate[:, :, 1] = self.R
        return nstate

    def next_step(self, state, a):
        coords = self.to_coords(state)
        movement = [[-1, 0], [0, 1], [1, 0], [0, -1]][a]
        new_coords = [coords[0] + movement[0], coords[1] + movement[1]]
        if not self.check_coord(new_coords):
            new_coords = coords
        reward = self.R[new_coords[0]][new_coords[1]]
        if reward == self.coin_value:
            self.R[new_coords[0]][new_coords[1]] = -0.1
        return self.to_state(new_coords), reward



if __name__ == '__main__':
    n_line, n_column = 3, 3
    mdp_env = MDP_environment(n_line, n_column)
    mdp_env.print_grid_infos()
    T = mdp_env.get_transition_matrix()
    R = mdp_env.get_reward_matrix()
    print(T.shape)
    print(R.shape)
