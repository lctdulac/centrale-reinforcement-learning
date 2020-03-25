#!/usr/bin/env python3
import os, sys
from testing_simulation import Simulation

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


sumoBinary = "/usr/share/sumo/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "./sumo_config/intersection.sumocfg"]

traci.start(sumoCmd)
step = 0

traci.close()
