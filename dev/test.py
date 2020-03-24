from MDP_environment import MDP_environment
from QL_Agent import AgentQ

n_lin, n_col = 6, 5
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
agent = AgentQ(gamma, lr, eps, episodes, dims, T, R)

agent.Q_Learning()

agent.plot_history()
