from copy            import copy
from random          import randint
from scipy           import stats
class Agent:
    
    def __init__(self, dims, T, R, position, policy = []):
        
        self.T = T
        self.R = R
        self.policy    = policy
        self.position  = position
        self.history   = []
        self.dims      = dims
        self.n_states  = dims[0] * dims[1]
        self.actions   =  {'up' :  [0.8,0.1,0.,0.1],
                        'right':[0.1,0.8,0.1,0.],
                        'down': [0.,0.1,0.8,0.1],
                        'left': [0.1,0.,0.1,0.8]}
        
        self.n_actions = len(list(self.actions.keys()))
    
    def goToNext(self):
        """
        Fonction qui change la case de l'agent selon la politique optimale 
        Variable qui se remplit : self.history [] 
        Output   : Null
        """
        action_recommande = self.policy[self.position]
        distribution_prochain_etat = self.T[action_recommande, self.position, :].tolist()
        
        probs, etats = [], []
        
        for j in range(len(distribution_prochain_etat)):
            if distribution_prochain_etat[j] != 0:
                probs.append(distribution_prochain_etat[j])
                etats.append(j)
        
        # prochain_etat = np.random.choice(etats, 1, probs)
        
        custm = stats.rv_discrete(name="custm", values=(etats, probs))
        
        prochain_etat = custm.rvs(size=1)
        
        #self.history.append( (self.position, prochain_etat[0], action_recommande) )
        self.history.append([self.position // self.dims[1], self.position % self.dims[1]])
        self.position = prochain_etat[0]
    
        #print("nouvelle position: ", self.position)

    def finalpos(self):
        self.history.append([self.position // self.dims[1], self.position % self.dims[1]])
        
        
    def getHistory(self):
        """
        Fonction qui retourne une liste des déplacements de L'agent
        """
        return self.history

    def value_iteration(self,discount):
        """
        Entraine L'agent avec l'algorithme de Value Iteration
        """
        U  = copy(self.R.flatten())
        U_ = copy(self.R.flatten())
        
        epsilon  = 10**-2
        DiffUtil = 10
        print("Le shape de la matrice de transition est : "+str(self.T.shape))
        print("Nombre d'actions : "+str(self.n_actions))
        print("Le nombre d'états : "+str(self.n_states))
        while DiffUtil > epsilon : 
            U = copy(U_)

            for state in range(self.n_states):
                U_[state] = self.R.flatten()[state] + discount * max([sum([self.T[a, state, j]*U[j] for j in range(self.n_states)]) for a in range(self.n_actions)])
            DiffUtil = max(abs(U - U_))
        
        self.U = U
        self.utilities_opt_policy()

    def utilities_opt_policy(self):
        """
        Fonction Intermediaire utilisée par Value Iteration pour
        donnée la politique optimale Sachant les Utilités des états
        """
        policy = [randint(0,self.n_actions - 1) for i in range(self.n_states)]
        for state in range(self.n_states):
            PU_List = [sum([self.T[a, state, j]*self.U[j] for j in range(self.n_states)]) for a in range(self.n_actions)]
            policy[state] = PU_List.index(max(PU_List))
        self.policy = policy

    def policy_iteration(discount):
        """
        Entraine L'agent avec l'algorithme de Policy Iteration
        """
        U = copy(self.R.flatten())
        policy = self.utilities_opt_policy()
        changed = True
        while changed == True:
            for state in range(self.n_states):
                U[state] = self.R.flatten()[state] + discount*sum([self.T[policy[state], state, j]*U[j] for j in range(self.n_states)])
            changed = False 
        
            for state in range(self.n_states):
                if max([sum([self.T[a, state, j]*U[j] for j in range(self.n_states)]) for a in range(self.n_actions)]) > sum([self.T[policy[state], state, j]*U[j] for j in range(self.n_states)]):
                    PU_List = [sum([self.T[a, state, j]*U[j] for j in range(self.n_states)]) for a in range(self.n_actions)]
                    policy[state] = PU_List.index(max(PU_List))
                    changed = True
        return policy

