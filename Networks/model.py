import time
import matplotlib.pyplot as plt

from Dataset.dataLoader import *
from Dataset.makeGraph import *

import numpy as np

from Networks.Architectures.basicNetwork import Net
from Networks.Architectures.unet import UNet
from Networks.Architectures.pspn import PSPNet
from Networks.Architectures.gctx_unet import GCTx_UNet

np.random.seed(2885)
import os
import copy

import torch
torch.manual_seed(2885)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import torch_optimizer

EPSILON_OVERFITTING = 0.05
EPSILON_ACCURACY = 0.01

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
        self.image_size    = param["EVALUATE"]["IMAGE_SIZE"]

        # -----------------------------------
        # NETWORK ARCHITECTURE INITIALISATION
        # -----------------------------------
        # self.model = Net(param).to(self.device)
        # self.model = PSPNet(param).to(self.device)
        self.model = UNet(param).to(self.device)
        # self.model = GCTx_UNet(param).to(self.device)
        # -------------------
        # TODO TRAINING PARAMETERS
        # -------------------
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch_optimizer.Adahessian(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch_optimizer.Yogi(self.model.parameters(), lr=self.lr)

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
    def loadWeights(self, epoch_number): 
        self.model.load_state_dict(torch.load(self.resultsPath + f'/_Weights/wghts_e{epoch_number}.pkl', weights_only=True))

    # -----------------------------------
    # TRAINING LOOP (fool implementation)
    # -----------------------------------
    def train(self):
        # TODO: You must write the loop to train and validate your model for a given number of epoch. At
        #  the end of the training, you are asked to print your train and validation loss curves into a graph.
        # train for a given number of epochs
        # for i in range(self.epoch):
        #    print("Loss at i-th epoch: ", str(np.random.random_sample()))
        #    modelWts = copy.deepcopy(self.model.state_dict())
        train_losses = []
        validations = []
        total_time = 0  # Initialize total time counter
        #self.loadWeights()
        for i in range(self.epoch):
            start_time = time.time()  # Start timing the epoch
            
            # Early stopping
            if i > 0 and train_losses[-1] < 2000.0 and validations[-1] < 2500.0 and \
                validations[-1] > train_losses[-1] + EPSILON_OVERFITTING and \
                validations[-2] - validations[-1] < EPSILON_ACCURACY:
                print(f"Early stopping caused by overfitting or no accuracy improvement.")
                print(f"Loss difference = {validations[-1] - train_losses[-1]}")
                print(f"Accuracy difference = {validations[-2] - validations[-1]}")
                break

            self.model.train(True)
            size_train = len(self.trainDataLoader)
            size_val = len(self.valDataLoader)
            train_loss = 0
            val_loss = 0
            for batch_idx, (images, masks, resizedImg) in enumerate(self.trainDataLoader):
                # Get images and associated mask
                images, masks = images.to(self.device), masks.to(self.device),

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # Make predictions for this batch
                predictions = self.model(images)
                predictions = predictions.squeeze(1)

                # Compute the loss and its gradients
                loss = self.criterion(predictions, masks)
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                train_loss += loss.item()
            train_loss /= size_train
            train_losses.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch_idx, (images, masks, resizedImg) in enumerate(self.valDataLoader):
                    images, masks = images.to(self.device), masks.to(self.device)
                    predictions = self.model(images)
                    predictions = predictions.squeeze(1)
                    loss = self.criterion(predictions, masks)
                    val_loss += loss.item()
                val_loss /= size_val
                validations.append(val_loss)

            # End timing the epoch
            epoch_time = time.time() - start_time
            total_time += epoch_time

            # Print learning curves
            # Implement this...
            print(f"Epoch {i + 1}/{self.epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            # Open the file in append mode and write the epoch information
            with open(os.path.join(self.resultsPath, 'res_epoch.txt'), 'a') as f:
                f.write(f"Epoch {i + 1}/{self.epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

            # Save the model weights
            modelWts = copy.deepcopy(self.model.state_dict())
            wghtsPath  = self.resultsPath + '/_Weights/'
            createFolder(wghtsPath)
            torch.save(modelWts, wghtsPath + f'/wghts_e{i+1}.pkl')

        # Print the average epoch time
        avg_time = total_time / self.epoch
        print(f"Average time per epoch: {avg_time:.2f} seconds")

        # plt.plot(range(1, self.epoch + 1), train_losses, label='Train Loss')
        # plt.plot(range(1, self.epoch + 1), validations, label='Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Train and Validation Loss Curves')
        # plt.legend()
        # plt.show()
        # plt.savefig(self.resultsPath + '/Plots/learning_curves.png')



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
        allInputs, allPreds, allGT, allPredsTresh = [], [], [], []
        for idx, (images, GT, resizedImg) in enumerate(self.testDataLoader):
            images      = images.to(self.device)
            predictions = self.model(images)

            images, predictions = images.to('cpu'), predictions.to('cpu')
            pred_masks = (predictions.detach().numpy() >= 0.5).squeeze(1)


            allInputs.extend(resizedImg.data.numpy())
            allPreds.extend(predictions.data.numpy())
            allGT.extend(GT.data.numpy())
            allPredsTresh.extend(pred_masks)

            # For now the score is just the delta between our prediction and the ground truth for each images
            gt_masks = GT.detach().numpy()
            scores = np.append(scores, np.sum(np.abs(gt_masks - pred_masks), axis=(1,2)).astype(int))
            dice_scores = np.append(dice_scores, [self.dice_coefficient(pred, gt) for pred, gt in zip(pred_masks, gt_masks)])
            iou_scores = np.append(iou_scores, [self.iou(pred, gt) for pred, gt in zip(pred_masks, gt_masks)])

        allInputs = np.array(allInputs)
        allPreds  = np.array(allPreds)
        allGT     = np.array(allGT)
        allPredsTresh = np.array(allPredsTresh)

        showPredictions(allInputs, allPreds, allGT, allPredsTresh, self.resultsPath)

        # Quantitative Evaluation
        mean_score = np.mean(scores)
        median_score = np.median(scores)

        total_pixels = self.image_size * self.image_size
        pixel_accuracies = [(score / total_pixels) * 100 for score in scores]
        mean_pixel_accuracy = 100.0 - np.mean(pixel_accuracies)
        median_pixel_accuracy = 100.0 - np.median(pixel_accuracies)
        std_pixel_accuracy = np.std(pixel_accuracies)

        mean_dice_score = np.mean(dice_scores)
        median_dice_score = np.median(dice_scores)
        std_dice_score = np.std(dice_scores)

        mean_iou_score = np.mean(iou_scores)
        median_iou_score = np.median(iou_scores)
        std_iou_score = np.std(iou_scores)

        # Display the results
        print(f'Mean score = {mean_score}\nMedian score = {median_score}')
        print(f'Mean Pixel Accuracy = {mean_pixel_accuracy}%\nMedian Pixel Accuracy = {median_pixel_accuracy}%\nSTD Pixel Accuracy = {std_pixel_accuracy}%')
        print(f'Mean dice score = {mean_dice_score}\nMedian dice score = {median_dice_score}\nSTD dice score = {std_dice_score}')
        print(f'Mean IoU score = {mean_iou_score}\nMedian IoU score = {median_iou_score}\nSTD IoU score = {std_iou_score}')
        
        # Write to the file in append mode
        with open(os.path.join(self.resultsPath, 'res_scores.txt'), 'a') as f:
            f.write(f"Mean score = {mean_score}\n")
            f.write(f"Median score = {median_score}\n")
            f.write(f"Mean Pixel Accuracy = {mean_pixel_accuracy:.2f}%\n")
            f.write(f"Median Pixel Accuracy = {median_pixel_accuracy:.2f}%\n")
            f.write(f"STD Pixel Accuracy = {std_pixel_accuracy:.2f}%\n")
            f.write(f"Mean dice score = {mean_dice_score}\n")
            f.write(f"Median dice score = {median_dice_score}\n")
            f.write(f"STD dice score = {std_dice_score}\n")
            f.write(f"Mean IoU score = {mean_iou_score}\n")
            f.write(f"Median IoU score = {median_iou_score}\n")
            f.write(f"STD IoU score = {std_iou_score}\n")
    
    def dice_coefficient(self, pred_mask, gt_mask):
        intersection = np.sum(pred_mask * gt_mask)
        return (2 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask))

    def iou(self, pred_mask, gt_mask):
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        return intersection / union
