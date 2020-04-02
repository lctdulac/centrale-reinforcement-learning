from Env.MDP_environment import MDP_environment
from Agents.AgentQL      import AgentQL
from Agents.AgentSARSA   import AgentSARSA
from Agents.AgentDQL     import AgentDQL
from Agents.AgentITL     import AgentITL
from GUI.GUI             import show_trajectory, draw_grid

import argparse
from time import time

""" This script allows the testing of the Q-learning, the SARSA or the DQL algorithm."""

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-gs", "--grid_size", type=int, nargs=2,
                    help='Number of line and colums for the grid')
parser.add_argument("-a", "--algorithm", type=str,
                    help='Name of the algo between (itlearning, qlearning, sarsa, deepq)')
parser.add_argument("-e", "--episodes", type=int,
                    help='Number of episodes.')
parser.add_argument("-d", "--display", type=int,
                    help='Display the maze and the mouse. 0 or 1.')

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


    gamma = 0.96
    lr = 0.1
    eps = 0.2
    episodes = args.episodes

    # Deep-Q Agent setup

    batch_size = 32
    train_every = 10
    update_every = 20

    # Training

    begin = time()

    if args.algorithm == "itlearning":
        agent = AgentITL([n_lin, n_col], mdp_env, 0, gamma)

    elif args.algorithm == "qlearning":
        agent = AgentQL(gamma, lr, eps, episodes, mdp_env)
 
    elif args.algorithm == "sarsa":
        agent = AgentSARSA(gamma, lr, eps, episodes, mdp_env)

    elif args.algorithm == "deepq":
        agent = AgentDQL(gamma, lr, eps, episodes, batch_size, train_every, update_every, mdp_env)

    else:
        print("Algorithm name not recognized.")

    agent.learning()

    # Results

    end = time()
    print("====> Total training time: %.2fs" % (end-begin))

    agent.plot_history()

    if args.display == 1:

        if args.algorithm == "itlearning":
            show_trajectory(mdp_env.n_lin,
                    mdp_env.n_col,
                    mdp_env.coins,
                    mdp_env.treasure,
                    mdp_env.traps,
                    mdp_env.obstacles,
                    agent.history)
        else:
            for epi in agent.episode_memory_buffer:
                show_traj(epi, mdp_env)
