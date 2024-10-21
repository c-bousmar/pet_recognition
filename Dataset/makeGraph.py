from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


# --------------------------------------------------------------------------------
# CREATE A FOLDER IF IT DOES NOT EXIST
# INPUT: 
#     - desiredPath (str): path to the folder to create
# --------------------------------------------------------------------------------
def createFolder(desiredPath): 
    if not os.path.exists(desiredPath):
        os.makedirs(desiredPath)


# --------------------------------------------------------------------------------
# DISPLAY A 5X5 IMAGES FROM A DATALOADER INSTANCE
# INPUTS: 
#     - dataLoader (Dataset): Instance of Dataset
#     - param (dic): dictionnary containing the parameters defined in the 
#                    configuration (yaml) file
# --------------------------------------------------------------------------------
def showDataLoader(dataLoader, param):
    cols, rows = 5, 5
    figure, ax = plt.subplots(nrows=rows, ncols=cols, dpi=280)
    row = 0
    for (imgBatch, maskBatch, _) in dataLoader:
        if row == rows: 
            break
        for col in range(cols): 
            img  = imgBatch[col].numpy().astype('uint8')
            mask = maskBatch[col].numpy().astype('uint8')
            ax[row, col].imshow(img.transpose((1, 2, 0)))
            ax[row, col].imshow(mask*255, interpolation="nearest", alpha=0.3, cmap='Oranges')
            ax[row, col].set_axis_off()
        row += 1 
    plt.tight_layout()
    plt.show()
    

# --------------------------------------------------------------------------------
# DISPLAY A SINGLE IMAGE AND GT MASK WITH THE CORRESPONDING PRECICTION 
# INPUTS: 
#     - img (arr): a 3D numpy array containing the image (ch x depth x width)
#     - pred (arr): a binary 2D numpy array containing the predicted mask
#                  (depth x width)
#     - GT (arr): a binary 2D numpy array containing the ground truth mask
#                  (depth x width)
#     - filePath (str): path to save the image file
# --------------------------------------------------------------------------------
def singlePrediction(img, pred, GT, filePath): 
    figure, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img.transpose((1, 2, 0)))
    ax[0].set_title("Input")
    ax[0].set_axis_off()

    ax[1].imshow(GT, interpolation="nearest")
    ax[1].set_title("GT")
    ax[1].set_axis_off()

    ax[2].imshow(np.squeeze(pred)*255, interpolation="nearest")
    ax[2].set_title("Prediction")
    ax[2].set_axis_off()

    plt.tight_layout()
    plt.savefig(filePath)
    plt.close()

# --------------------------------------------------------------------------------
# DISPLAY ALL IMAGES AND PREDICTIONS
# INPUTS: 
#     - allInputs (list): list of 3D numpy arrays containing the input images 
#     - allPreds (list): list of 2D numpy arrays containing the predicted masks
#     - allGT (list): list of 2D numpy arrays containing the ground truth masks 
#     - resultPath (str): path to folder in which to save the image files
# --------------------------------------------------------------------------------
def showPredictions(allInputs, allPreds, allGT, resultPath):
    idx = 0
    for (img, pred, GT) in zip(allInputs, allPreds, allGT): 
        filePath = os.path.join(resultPath, "Test", str(idx))
        createFolder(os.path.join(resultPath, "Test"))
        singlePrediction(img, pred, GT, filePath)
        if idx > 30: 
            break
        idx += 1

    