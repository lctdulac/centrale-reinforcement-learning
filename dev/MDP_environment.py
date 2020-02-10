import numpy as np
from random import random, randint

class MDP_environment():

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
            i, j = randint(0, self.n_lin), randint(0,self.n_col)
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
            # Si on est sur un état adjacent, et que cet état n'est pas obstrué, on met un 1 qui signifie qu'on peut aller sur cet état à partir
            # de l'état où l'on est
                if (j==i-self.n_col or j==i+1 or j==i+self.n_col or j==i-1) and self.grid[j//self.n_col,j%self.n_col]!=-1 :
                    adj[i,j] = 1
        #we can now contruct the transition matrix T
        T = []
        rotation_vect = [-self.n_col, 1,self.n_col, -1]
        for action in self.actions:
            Ta = np.zeros((self.n_states, self.n_states))
            prob_vector = self.actions[action]
            for i in range(self.n_states) :
                for k in range(len(rotation_vect)):
                    j = i+rotation_vect[k]
                    if 0<=j<=15:
                        Ta[i, j] = adj[i, j]*prob_vector[k]
                Lsum = np.sum(Ta[i,:])
                if Lsum !=1:
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
    n_line = 5
    n_column = 6
    mdp_env = MDP_environment(n_line, n_column)
    mdp_env.print_grid_infos()
    T = mdp_env.get_transition_matrix()
    print(T.shape)
