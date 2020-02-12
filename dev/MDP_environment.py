import numpy as np
from random import random, randint, seed

#set the seed so we generate the same grid
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
        self.make_grid(p_obs, n_traps, p_coins)
        self.actions = {'up' :  [0.8,0.1,0.,0.1],
                        'right':[0.1,0.8,0.1,0.],
                        'down': [0.,0.1,0.8,0.1],
                        'left': [0.1,0.,0.1,0.8]}
        self.set_rewards_matrix()

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
                if i!=0 and j!=0:
                    p = random()
                    if p < p_obs:
                        self.grid[i, j] = -1
                        self.obstacles.append((i,j))

    def set_treasure(self):
        self.treasure = 0
        while self.treasure == 0:
            i, j = randint(self.n_lin-2, self.n_lin), randint(self.n_col//2,self.n_col)
            if (i,j) not in self.obstacles:
                self.treasure = (i,j)

    def set_traps(self,n_traps):
        self.traps = []
        while n_traps>0:
            i, j = randint(0, self.n_lin - 1), randint(0,self.n_col - 1)
            if (i,j) not in self.obstacles and (i,j)!=self.treasure:
                self.traps.append((i,j))
                n_traps -= 1

    def set_coins(self,p_coins):
        self.coins = []
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if (i,j) not in self.obstacles and (i,j) not in self.traps and (i,j)!=self.treasure:
                    p = random()
                    if p < p_coins:
                        self.coins.append((i,j))

    def get_transition_matrix(self):
        self.set_transition_matrix()
        return self.T

    def set_transition_matrix(self):
        #we first create and adjacent matrix to ease the construction of
        #the transition matrix
        adj = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states) :
            for j in range(self.n_states) :
                if (j==i-self.n_col or j==i+1 or j==i+self.n_col or j==i-1) and self.grid[j//self.n_col,j%self.n_col]!=-1 :
                    adj[i,j] = 1
        #we can now contruct the transition matrix T
        T = []
        rotation_vect = [-self.n_col, 1,self.n_col, -1]
        for action in self.actions:
            Ta = np.zeros((self.n_states, self.n_states))
            prob_vector = self.actions[action]
            for i in range(self.n_states) :
                if np.sum(adj[i,:])!=1.0:
                    for k in range(len(prob_vector)):
                        j = i+rotation_vect[k]
                        if 0<=j<self.n_states:
                            Ta[i, j] = adj[i, j]*prob_vector[k]
                elif np.sum(adj[i,:])==1.0:
                    Ta[i,:] = adj[i,:]
                Lsum = np.sum(Ta[i,:])
                if Lsum!=1:
                    Ta[i,:]*=1/Lsum
            T.append(Ta)
        self.T = np.array(T)

    def set_rewards_matrix(self):
        R = np.zeros((self.n_lin, self.n_col))
        for i in range(self.n_lin):
            for j in range(self.n_col):
                if (i,j) in self.obstacles:
                    reward = 0
                elif (i,j) in self.traps :
                    reward = -1
                elif (i,j) == self.treasure :
                    reward = 1
                elif (i,j) in self.coins :
                    reward = 0.2
                else :
                    reward = -0.1
                R[i,j] = reward
        self.R = R

    def print_grid_infos(self):
        print('Grid : \n', self.grid)
        print('\nTreasure coordinate : \n', self.treasure)
        print('\nObstacles coordinate : \n', self.obstacles)
        print('\nTraps coordinate : \n', self.traps)
        print('\nCoins coordinate : \n', self.coins)


if __name__ == '__main__':
    n_line, n_column = 5, 6
    mdp_env = MDP_environment(n_line, n_column)
    mdp_env.print_grid_infos()
    T = mdp_env.get_transition_matrix()
    print(T.shape)