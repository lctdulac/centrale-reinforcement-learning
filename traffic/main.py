#!/usr/bin/env python3
from config import setup
setup()
# Importation Bibliothèques Traci 
import traci
from training_simulation import Simulation
from generator import TrafficGenerator



sumoBinary = "/usr/share/sumo/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "./sumo_config/intersection.sumocfg"]




if __name__ == "__main__":
    gen = TrafficGenerator(1600,300)
    Sim = Simulation(TrafficGen = gen,Model=[],sumo_cmd = sumoCmd, max_steps = 1600, green_duration=10, yellow_duration=4, num_states=80, num_actions=4)
    
