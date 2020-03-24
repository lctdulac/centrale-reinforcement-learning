# Guide mise en place environnement SUMO 

Cette démarche est inspiré du Papier scientifique : A Deep Reinforcement Learning Approach to Adaptive Traffic Lights Management.\\
Lien : http://ceur-ws.org/Vol-2404/paper07.pdf

## 1. Prérequis 

Pour pouvoir développer cette partie il faut installer les outils suivants : 

* Python 3.7
* SUMO ( Tutoriel Installation : https://sumo.dlr.de/docs/Installing.html#linux )
* Traci - API Python pour communiquer avec le simulateur de traffic ( https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html )

## 2. Lancement du simulateur

SUMO peut être lancé en 2 mode ( Mode graphique et Mode serveur ). Avant le lancement de n'importe quelle simulation il est nécessaire
de définir certaines entrées : \\

* Un fichier de configuration central (.sumocfg)
* Le fichier XML contenant la configuration du réseau routier à utiliser
* Le ficher  XML contenant la configuration des véhicules , flux de déplacement et feux

Pour lancer SUMO ( sur Windows / Linux ) il suffit de lancer la commande suivante : \\

```bash
sumo -c ./sumo_config/intersection.sumocfg
```

ou en mode graphique 

```bash
sumo-gui -c ./sumo_config/intersection.sumocfg
```


## 3. Arborescence du projet 



## 4. Tests lecture écriture

Dans cette partie nous testons la connectivité entre SUMO - Python

```bash
python3 test_connect.py
```

Il faut s'attendre après le test à voir une position de voiture qui bouge.

