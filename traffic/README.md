# SUMO environment setup guide 

This has been adapted from the research paper : A Deep Reinforcement Learning Approach to Adaptive Traffic Lights Management.\\
Link : http://ceur-ws.org/Vol-2404/paper07.pdf

## 1. Requirement 

Inorder to use this code you need to install the following : 

* Python 3.7
* Tensorflow 
* SUMO ( Installation tutorial : https://sumo.dlr.de/docs/Installing.html#linux )
* Traci - Python API to connect to the trafic simulator ( https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html )

## 2. Using the simulator

SUMO can be lauched in two different modes : GUI and server. Before starting simulation, you will need to define some entry variables : \\

* a central configuration file (.sumocfg)
* an XML file with the configuration of the road network 
* an XML file with the simulation configurations : vehicules and traffic lights

To Launch SUMO ( Windows / Linux ) you will need to use to following command lines : \\

```bash
sumo -c ./sumo_config/intersection.sumocfg
```

or with GUI : \\

```bash
sumo-gui -c ./sumo_config/intersection.sumocfg
```

## 3. Repository usage

You need to have all the requirement installed to run these command.

In order to use the repository you will have to setup the parameters in the ```training_settings.ini``` file.

Then simply use the following command line :
```bash
python3 training_main.py
```

After this a new folder with the model will be created. 

You can use the saved model to run a test : write the parameters you want in the ```testing_settings.ini``` file and then use :
```bash
python3 testing_main.py
```
A test file with plots will be created in the model folder.

This code as been adapted from the following repository : https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control/tree/master/TLCS .

