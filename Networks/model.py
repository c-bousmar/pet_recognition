from Dataset.dataLoader import *
from Dataset.makeGraph import *
from Networks.Architectures.basicNetwork import *

import numpy as np
np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


######################################################################################
#
# CLASS DESCRIBING THE INSTANTIATION, TRAINING AND EVALUATION OF THE MODEL 
# An instance of Network_Class has been created in the main.py file
# 
######################################################################################

class Network_Class: 
    # --------------------------------------------------------------------------------
    # INITIALISATION OF THE MODEL
    # INPUTS: 
    #     - param (dic): dictionnary containing the parameters defined in the 
    #                    configuration (yaml) file
    #     - imgDirectory (str): path to the folder containing the images 
    #     - maskDirectory (str): path to the folder containing the masks
    #     - resultsPath (str): path to the folder containing the results of the 
    #                          experiement
    # --------------------------------------------------------------------------------
    def __init__(self, param, imgDirectory, maskDirectory, resultsPath):
        # ----------------
        # USEFUL VARIABLES 
        # ----------------
        self.imgDirectory  = imgDirectory
        self.maskDirectory = maskDirectory
        self.resultsPath   = resultsPath
        self.epoch         = param["TRAINING"]["EPOCH"]
        self.device        = param["TRAINING"]["DEVICE"]
        self.lr            = param["TRAINING"]["LEARNING_RATE"]
        self.batchSize     = param["TRAINING"]["BATCH_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        self.model = Net(param).to(self.device)

        # -------------------
        # TODO TRAINING PARAMETERS
        # -------------------
        self.criterion = ...
        self.optimizer = ... 

        # ----------------------------------------------------
        # DATASET INITIALISATION (from the dataLoader.py file)
        # ----------------------------------------------------
        self.dataSetTrain    = OxfordPetDataset(imgDirectory, maskDirectory, "train", param)
        self.dataSetVal      = OxfordPetDataset(imgDirectory, maskDirectory, "val",   param)
        self.dataSetTest     = OxfordPetDataset(imgDirectory, maskDirectory, "test",  param)
        self.trainDataLoader = DataLoader(self.dataSetTrain, batch_size=self.batchSize, shuffle=True,  num_workers=4)
        self.valDataLoader   = DataLoader(self.dataSetVal,   batch_size=self.batchSize, shuffle=False, num_workers=4)
        self.testDataLoader  = DataLoader(self.dataSetTest,  batch_size=self.batchSize, shuffle=False, num_workers=4)


    # ---------------------------------------------------------------------------
    # LOAD PRETRAINED WEIGHTS (to run evaluation without retraining the model...)
    # ---------------------------------------------------------------------------
    def loadWeights(self): 
        self.model.load_state_dict(torch.load(self.resultsPath + '/_Weights/wghts.pkl'))

    # -----------------------------------
    # TRAINING LOOP (fool implementation)
    # -----------------------------------
    def train(self):
        # TODO: You must write the loop to train and validate your model for a given number of epoch. At
        #  the end of the training, you are asked to print your train and validation loss curves into a graph.
        # train for a given number of epochs
        for i in range(self.epoch):
            print("Loss at i-th epoch: ", str(np.random.random_sample()))
            modelWts = copy.deepcopy(self.model.state_dict())

        # Print learning curves
        # Implement this...

        # Save the model weights
        wghtsPath  = self.resultsPath + '/_Weights/'
        createFolder(wghtsPath)
        torch.save(modelWts, wghtsPath + '/wghts.pkl')



    # -------------------------------------------------
    # EVALUATION PROCEDURE (ultra basic implementation)
    # -------------------------------------------------
    def evaluate(self):
        self.model.train(False)
        self.model.eval()

        scores = np.array([])
        dice_scores = np.array([])
        iou_scores = np.array([])

        
        # Qualitative Evaluation 
        allInputs, allPreds, allGT = [], [], []
        for idx, (images, GT, resizedImg) in enumerate(self.testDataLoader):
            images      = images.to(self.device)
            predictions = self.model(images)

            images, predictions = images.to('cpu'), predictions.to('cpu')

            allInputs.extend(resizedImg.data.numpy())
            allPreds.extend(predictions.data.numpy())
            allGT.extend(GT.data.numpy())

            # For now the score is just the delta between our prediction and the ground truth for each images
            pred_masks = (predictions.detach().numpy() >= 0.5).squeeze(1)
            gt_masks = GT.detach().numpy()
            scores = np.append(scores, np.sum(np.abs(gt_masks - pred_masks), axis=(1,2)).astype(int))
            dice_scores = np.append(dice_scores, [self.dice_coefficient(pred, gt) for pred, gt in zip(pred_masks, gt_masks)])
            iou_scores = np.append(iou_scores, [self.iou(pred, gt) for pred, gt in zip(pred_masks, gt_masks)])

        allInputs = np.array(allInputs)
        allPreds  = np.array(allPreds)
        allGT     = np.array(allGT)

        #showPredictions(allInputs, allPreds, allGT, self.resultsPath)

        # Quantitative Evaluation
        print(f'Mean score = {np.mean(scores)}\nMedian score = {np.median(scores)}')
        print(f'Mean dice score = {np.mean(dice_scores)}\nMedian dice score = {np.median(dice_scores)}')
        print(f'Mean iou score = {np.mean(iou_scores)}\nMedian iou score = {np.median(iou_scores)}')
    
    def dice_coefficient(self, pred_mask, gt_mask):
        intersection = np.sum(pred_mask * gt_mask)
        return (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask))

    def iou(self, pred_mask, gt_mask):
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        return intersection / union
