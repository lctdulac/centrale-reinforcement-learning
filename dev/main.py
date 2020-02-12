import argparse
from MDP_environment import MDP_environment
from GUI import show_trajectory

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
    #train agent on this environment
    trajectory = []
    #show results
    show_trajectory(mdp_env.n_lin, 
                    mdp_env.n_col, 
                    mdp_env.coins, 
                    mdp_env.treasure, 
                    mdp_env.traps, 
                    mdp_env.obstacles, 
                    trajectory)
