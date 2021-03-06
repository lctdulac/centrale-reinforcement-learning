# Reinforcement Learning
#### April 2020
> Authors : Théo Hiraclides | Ravi Hassanaly | Mathieu Perrin | Adam Mabrouk | Lancelot Prégniard

_This repository contains the code of our final year project as part of the Master of Engineering at École Centrale de Lyon (France)._

The goals were :
- to implement several reinforcement learning algorithms.
- to develop a test environment for reinforcement learning.
- to apply a state-of-the-art reinforcement learning algorithm to a real world problem.


## Installation

Please install all the libraries described in `requirements.txt` in your python environment.

## Architecture

The `dev` folder contains the code for :
- `Env`: The Markov Decision Process environment of the "mouse in a maze" game.
- `Agents` : The code for the Iteration Learning (Value-Iteration, Policy-Iteration), Q-learning, SARSA and Deep Q-learning algorithms.
- `GUI` : The code for the PyGame Graphical Interface.

The `research` folder contains our first Jupyter Notebooks. We used it as a starting point in our project.

The `traffic` folder contains the code for the Deep Q-learning implementation for <b>Traffic Control</b>, a real-world problem simulated in the <b>SUMO</b> environment. It simulates the management of an intersection with traffic lights and cars.


## Usage example

#### Mouse in a Maze (test environment)

Run the `letsplay.py` to test any algorithm in this environment.

The arguments are :
- `-gs` or `--grid_size` : size of the grid as two integers
- `-a` or `--algorithm` : name of the algorithm between (itlearning, qlearning, sarsa, deepq)
- `-e` or `--episodes` : number of episodes for the simulation as an integer
- `-d` or `--display` : (0, 1) choice to display the last episodes in the pygame GUI or not


Examples:
```
python letsplay.py -gs 6 6 -a sarsa -e 1000 -d 1
python letsplay.py -gs 6 6 -a deepq -e 100 -d 0
python letsplay.py -gs 8 8 -a itlearning -e 100 -d 1
```


#### Traffic control (SUMO environment)

_The README.md can be found in the `traffic` folder <a href="https://github.com/lctdulac/centrale-reinforcement-learning/tree/master/traffic">here.</a>_
