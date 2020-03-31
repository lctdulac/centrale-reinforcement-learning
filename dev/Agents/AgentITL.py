from copy import copy
from random import randint
from scipy import stats
import random


class AgentITL:
    """
    In this class we create a Policy / Value Learning Agent, 
    running in a stochastic MDP environment.
    It uses either policy / value learning .
    ________________________________________________________________
    Attributes :
    We consider m the number of lines and n number of columns
    - T        :  np.array (4,m,n)
    The transition matrix of the MDP
    - R        :  np.array (m,n)
    The reward matrix
    - discound : int
    The discount factor
    - policy   :  list (16)
    The policy
    - position :  list [x,y]
    The agent's position  
    - history  :  list []
    A list of old mouvement   
    - dims     : tuple (m,n)
    The dimensions of the matrix    
    - n_states : m * n 
    The number of existing states
    - actions  : dict()
     A dictionnary mapping the actions to the probabilities of ending in a specific direction
    """
    def __init__(self, dims, mdp_env, position, discount, policy=[]):
        
        self.T = mdp_env.T
        self.R = mdp_env.R
        self.policy = policy
        self.position = position
        self.discount  = discount
        self.history = []
        self.dims = dims
        self.n_states = dims[0] * dims[1]
        self.actions = {'up':  [0.8, 0.1, 0., 0.1],
                        'right': [0.1, 0.8, 0.1, 0.],
                        'down': [0., 0.1, 0.8, 0.1],
                        'left': [0.1, 0., 0.1, 0.8]}

        self.n_actions = len(list(self.actions.keys()))
        

    def goToNext(self):
        """
        Method that changes the Agent's state given the optimal policy
        Result is stores in  : self.history [] 
        """
        action_recommande = self.policy[self.position]
        distribution_prochain_etat = self.T[action_recommande, self.position, :].tolist()
        probs, etats = [], []

        for j in range(len(distribution_prochain_etat)):
            if distribution_prochain_etat[j] != 0:
                probs.append(distribution_prochain_etat[j])
                etats.append(j)

        custm = stats.rv_discrete(name="custm", values=(etats, probs))
        prochain_etat = custm.rvs(size=1)
        self.history.append([self.position // self.dims[1],
                             self.position % self.dims[1]])
        self.position = prochain_etat[0]

    def finalpos(self):
        self.history.append([self.position // self.dims[1],
                             self.position % self.dims[1]])

    def getHistory(self):
        """
        Returns the history of the Agent
        """
        return self.history

    

    def value_iteration(self):
        """
        Trains the Agent with Value Iteration Algorithm
        """
        U = copy(self.R.flatten())
        U_ = copy(self.R.flatten())

        epsilon = 10**-2
        DiffUtil = 10
        print("Le shape de la matrice de transition est : "+str(self.T.shape))
        print("Nombre d'actions : "+str(self.n_actions))
        print("Le nombre d'états : "+str(self.n_states))
        while DiffUtil > epsilon:
            U = copy(U_)

            for state in range(self.n_states):
                U_[state] = self.R.flatten()[state] + self.discount * max([sum([self.T[a, state, j]*U[j]
                                                                           for j in range(self.n_states)]) for a in range(self.n_actions)])
            DiffUtil = max(abs(U - U_))

        self.U = U
        self.utilities_opt_policy()

    def utilities_opt_policy(self):
        """
        Intermediate function to calculate the optimal policy
        given the utility of all states.
        """
        policy = [randint(0, self.n_actions - 1) for i in range(self.n_states)]
        for state in range(self.n_states):
            PU_List = [sum([self.T[a, state, j]*self.U[j]
                            for j in range(self.n_states)]) for a in range(self.n_actions)]
            policy[state] = PU_List.index(max(PU_List))
        self.policy = policy

    def policy_iteration(self):
        """
        Trains the agent with the Policy learning algorithm
        """
        U       = copy(self.R.flatten())
        policy  = [random.randint(0,3) for i in range(self.n_states)] 
        changed = True
        while changed == True:
            for state in range(self.n_states):
                U[state] = self.R.flatten()[state] + self.discount * \
                    sum([self.T[policy[state], state, j]*U[j]
                         for j in range(self.n_states)])
            changed = False

            for state in range(self.n_states):
                if max([sum([self.T[a, state, j]*U[j] for j in range(self.n_states)]) for a in range(self.n_actions)]) > sum([self.T[policy[state], state, j]*U[j] for j in range(self.n_states)]):
                    PU_List = [sum([self.T[a, state, j]*U[j] for j in range(self.n_states)])
                               for a in range(self.n_actions)]
                    policy[state] = PU_List.index(max(PU_List))
                    changed = True
        self.policy = policy

    def learning(self):
        print("What type of iteration algorithm to use :")
        print("[0] Value Iteration")
        print("[1] Policy Iteration")
        choice = int(input())
        if choice == 0:
            self.value_iteration()
        elif choice == 1:
            self.policy_iteration()
        else: 
            print("[-] Wrong choice try again ....")
            raise

    def plot_history(self, grid=False):
        print("No plot for this type of algorithm")
