import torch
import torch.nn as nn  # neural network
import torch.nn.functional as F

#==============================================================
#  Two convolutional layers and Two fully connected layers
#===============================================================

class Fashion_MNIST_Network(nn.Module):
    '''
    Arguments:
    in_channels  – Number of channels in the input image.
    out_channels – Number of channels produced by the convolution.
    kernel_size – Size of the convolving kernel.
    stride – Stride of the convolution
    '''
    def __init__(self):
        super(Fashion_MNIST_Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=7*7*16, out_features=392)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=392, out_features=10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(-1,7*7*16)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x