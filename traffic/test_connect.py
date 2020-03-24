#!/usr/bin/env python3
import os, sys

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

while step < 1000:
   traci.simulationStep()
   position = traci.vehicle.getPosition('vehicle_0')
   print("Position du véhicule : "+str(position))
   step += 1

traci.close()
