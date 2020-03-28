from GUI import show_trajectory
from MDP_environment import MDP_environment
from random import randint

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

walk_len = 200
walk = [0]

while len(walk) < walk_len:
    a = randint(0, 3)
    walk.append(mdp_env.next_step(walk[-1], a)[0])

print(walk)

show_traj(walk, mdp_env)

