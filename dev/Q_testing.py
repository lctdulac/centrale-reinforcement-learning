from Env.MDP_environment import MDP_environment
from Agents.AgentQL import AgentQL
from Agents.AgentSARSA import AgentSARSA
from GUI.GUI import show_trajectory, draw_grid

import argparse
from time import time

""" This script allows the testing of the Q-learning, the SARSA or the DQL algorithm."""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-gs", "--grid_size", type=int, nargs=2,
                    help='Number of line and colums for the grid')
parser.add_argument("-a", "--algorithm", type=str,
                    help='Name of the algo between (qlearning, sarsa, deepq)')
parser.add_argument("-e", "--episodes", type=int,
                    help='Number of episodes.')
parser.add_argument("-d", "--display", type=bool,
                    help='Display the maze and the mouse. True or False.')

args = parser.parse_args()


def show_traj(states, mdp_env):
    trajectory = []
    for i in states:
        trajectory.append([i // n_col, i % n_col])
    show_trajectory(mdp_env.n_lin,
                    mdp_env.n_col,
                    mdp_env.coins,
                    mdp_env.treasure,
                    mdp_env.traps,
                    mdp_env.obstacles,
                    trajectory)


if __name__ == '__main__':

    print("Script arguments: ", args)

    # Environment setup

    [n_lin, n_col] = args.grid_size
    mdp_env = MDP_environment(n_lin, n_col)

    mdp_env.print_grid_infos()

    # Agent setup

    gamma = 0.9
    lr = 0.1
    eps = 0.2
    episodes = args.episodes

    # Training

    begin = time()

    if args.algorithm == "qlearning":
        agent = AgentQL(gamma, lr, eps, episodes, mdp_env)
 
    if args.algorithm == "sarsa":
        agent = AgentSARSA(gamma, lr, eps, episodes, mdp_env)

    if args.algorithm == "deepq":
        agent = AgentDQL(gamma, lr, eps, episodes, mdp_env)

    else:
        print("Algorithm name not recognized.")

    agent.learning()

    # Results

    end = time()
    print("====> Total training time: %.2f" % (end-begin))

    agent.plot_history()

    if args.display == True:
        for epi in agent.episode_memory_buffer:
            show_traj(epi, mdp_env)
