from MDP_environment import MDP_environment
from DQL_Agent import AgentDQ
from GUI import show_trajectory

from time import time


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


n_lin, n_col = 6, 6
mdp_env = MDP_environment(n_lin, n_col)

# show environment info
mdp_env.print_grid_infos()

# agent

gamma = 0.95
lr = 0.1
eps = 0.2
episodes = 300
dims = [n_lin, n_col]
T = mdp_env.T
R = mdp_env.R

print("R:", R)


begin = time()
agent = AgentDQ(gamma, lr, eps, episodes, dims, mdp_env)
agent.DQ_Learning()

agent.plot_history()

end = time()

print("Total time elapsed: %.2f" % (end-begin))

# SARSA : 31.16s pour 2000 steps
# Q-learning : 51.43 pour 2000 steps

for epi in agent.episode_memory_buffer:
    show_traj(epi, mdp_env)
