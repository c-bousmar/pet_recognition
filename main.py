from Networks.model import *
from Dataset.makeGraph import *

import argparse
import yaml

import os
from os.path import dirname, abspath
from termcolor import colored


rootDirectory    = dirname(abspath(__file__))
datasetDirectory = os.path.join(rootDirectory,    "Dataset")
imgDirectory     = os.path.join(datasetDirectory, "images")
maskDirectory    = os.path.join(datasetDirectory, "annotations", "trimaps")

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='DefaultExp')


######################################################################################
#
# MAIN PROCEDURE 
# launches an experiment whose parameters are described in a yaml file  
# 
# Example of use in the terminal: python main.py -exp DefaultExp
# with 'DefaultExp' beeing the name of the yaml file (in the Todo_list folder) with 
# the wanted configuration 
# 
######################################################################################

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open('Todo_List/' + parser.exp + '.yaml', 'r')
    param  = yaml.safe_load(stream)
    # Path to the folder that will contain results of the experiment 
    resultsPath = os.path.join(rootDirectory, "Results", parser.exp)

    # ------------------------
    # 1. NETWORK INSTANTIATION 
    # ------------------------
    myNetwork  = Network_Class(param, imgDirectory, maskDirectory, resultsPath)

    # ------------------------------------------ 
    # 2. VISUALISATION OF THE DATASET (OPTIONAL)
    # ------------------------------------------
    # Comment line below to skip the visualisation
    showDataLoader(myNetwork.trainDataLoader, param)

    # ------------------
    # 3. TRAIN THE MODEL  
    # ------------------
    print(colored('Start to train the network', 'red'))
    myNetwork.train()
    print(colored('The network is trained', 'red'))
    
    # ---------------------
    # 4. EVALUATE THE MODEL  
    # ---------------------  
    myNetwork.loadWeights()
    myNetwork.evaluate()
    

if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)

