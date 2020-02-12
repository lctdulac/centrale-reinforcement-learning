import argparse
from MDP_environment import MDP_environment
from MDP_Agent       import Agent
from GUI             import show_trajectory


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-gs", "--grid_size", type=int, nargs=2,
                    help='Number of line and colums for the grid')

args = parser.parse_args()

if __name__ == '__main__':
    #create environment
    [n_lin, n_col] = args.grid_size
    mdp_env = MDP_environment(n_lin, n_col)
    #show environment info
    mdp_env.print_grid_infos()
    mdp_env.set_transition_matrix()
    #Création et entrainement de l'Agent sur L'environnement
    A = Agent([n_lin, n_col], mdp_env.T, mdp_env.R, 0)
    
    A.value_iteration(0.96)
    print(A.policy)

    while A.R.flatten()[A.position] != 1:
        A.goToNext()
    print(A.history)  
    trajectory = A.history
    #show results
    show_trajectory(mdp_env.n_lin, 
                    mdp_env.n_col, 
                    mdp_env.coins, 
                    mdp_env.treasure, 
                    mdp_env.traps, 
                    mdp_env.obstacles, 
                    trajectory)
