from MDP_environment import MDP_environment
from QL_Agent import AgentQ
from GUI import show_trajectory

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

gamma = 0.9
lr = 0.1
eps = 0.2
episodes = 1000
dims = [n_lin, n_col]
T = mdp_env.T
R = mdp_env.R

print("R:", R)

agent = AgentQ(gamma, lr, eps, episodes, dims, T, R)

agent.Q_Learning()

agent.plot_history()

for eps in agent.episode_memory_buffer:
    show_traj(eps, mdp_env)