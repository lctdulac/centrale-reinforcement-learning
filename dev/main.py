import argparse
from MDP_environment import MDP_environment

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-gs", "--grid_size", type=int, nargs=2,
                    help='Number of line and colums for the grid')

args = parser.parse_args()

if __name__ == '__main__':
    [n_lin, n_col] = args.grid_size
    mdp_env = MDP_environment(n_lin, n_col)
    mdp_env.print_grid_infos()
