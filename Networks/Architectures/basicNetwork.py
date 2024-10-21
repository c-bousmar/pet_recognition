import torch.nn as nn
import torch.nn.functional as F

######################################################################################
#
# CLASS DESCRIBING A FOOL MODEL ARCHITECTURE
# An instance of Net has been created in the model.py file
# 
######################################################################################

class Net(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.nb_channel = param["MODEL"]["NB_CHANNEL"]
        self.conv1 = nn.Conv2d(3, self.nb_channel, 5, padding='same')
        self.conv2 = nn.Conv2d(self.nb_channel, self.nb_channel, 5, padding='same')
        self.conv3 = nn.Conv2d(self.nb_channel, self.nb_channel, 5, padding='same')
        self.conv4 = nn.Conv2d(self.nb_channel, 1, 5, padding='same')
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
