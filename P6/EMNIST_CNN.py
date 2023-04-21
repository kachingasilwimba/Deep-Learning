# https://github.com/machinecurve/extra_keras_datasets 
# https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

#========================================================================
# Two convolutional layers and two fully connected layers
#=========================================================================

# class EMNISTCNN(nn.Module):
#     '''
#     Arguments:
#     in_channels  – Number of channels in the input image.
#     out_channels – Number of channels produced by the convolution.
#     kernel_size – Size of the convolving kernel.
#     stride – Stride of the convolution
#     padding – Padding added to all four sides of the input.
#     '''
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(1, 20, stride = 1, kernel_size = 5)  # makes 20 maps of 24x24
    #     self.pool = nn.MaxPool2d(2, 2)                              # 20 maps of 12x12
    #     self.conv2 = nn.Conv2d(20, 40, stride = 1, kernel_size = 5) # 40 maps of 8x8
    #     self.pool = nn.MaxPool2d(2, 2)                              # 40 maps of 4x4
    #     self.fc1 = nn.Linear(40 * 4 * 4, 100)                       # flatten to 40*4*4 neurons                   
    #     self.fc2 = nn.Linear(100, 100)                              
        
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    # def __init__(self, fmaps1, fmaps2, dense, dropout):
    #     super(EMNISTCNN, self).__init__()
    #     self.conv1 = nn.Sequential(         
    #         nn.Conv2d(in_channels=1, out_channels=fmaps1, kernel_size=5, stride=1, padding='same'),                              
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #     )
    #     self.conv2 = nn.Sequential(         
    #         nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding='same'),                              
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #     )
    #     self.fcon1 = nn.Sequential(nn.Linear(49*fmaps2, dense), nn.ReLU())
    #     self.fcon2 = nn.Linear(dense, 47)
    #     self.dropout = nn.Dropout(p=dropout)
    
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = torch.flatten(x, 1)
    #     # x = x.view(x.size(0), -1)
    #     x = self.dropout(self.fcon1(x))
    #     x = self.fcon2(x)
    #     return x

    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Sequential(nn.Conv2d(1, 50, kernel_size=3, padding=1), 
    #                               nn.BatchNorm2d(50),
    #                               nn.ReLU(inplace=True)) #32*28*28
    #     self.conv2 = nn.Sequential(nn.Conv2d(50, 64, kernel_size=3, padding=1), 
    #                               nn.BatchNorm2d(64),
    #                               nn.ReLU(inplace=True),
    #                               nn.MaxPool2d(2)) #64*14*14
    #     self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
    #                               nn.BatchNorm2d(128),  
    #                               nn.ReLU(inplace=True)) #128*14*14
    #     self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
    #                               nn.BatchNorm2d(256), 
    #                               nn.ReLU(inplace=True),
    #                               nn.MaxPool2d(2)) #256*7*7
    #     self.classifier = nn.Sequential(nn.Flatten(),
    #                       nn.Linear(256*7*7, 1024),
    #                       nn.ReLU(),
    #                       nn.Linear(1024, 256),
    #                       nn.ReLU(),
    #                       nn.Linear(256, 47)
    #     )

    # def forward(self, xb):
    #     out = self.conv1(xb)
    #     out = self.conv2(out)
    #     out = self.conv3(out)
    #     out = self.conv4(out)
    #     out = self.classifier(out)
    #     return out
# =======================================
# class EMNISTCNN(nn.Module):

#     def __init__(self):
#         super(EMNISTCNN, self).__init__()
        
#         # 1 input image channel (grayscale), 10 output channels/feature maps
#         # 3x3 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(64,70,3, padding=1)
#         self.fc1 = nn.Linear(7*7*70,128)
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(128,47)
#         #self.fc3 = nn.Linear(128,64)
#         #self.fc4 = nn.Linear(64,47)
        
#         ## TODO: Define the rest of the layers:
#         # include another conv layer, maxpooling layers, and linear layers
#         # also consider adding a dropout layer to avoid overfitting


#     ## TODO: define the feedforward behavior
#     def forward(self, x):
#         # one activated conv layer
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         #x = self.dropout(x)
#         #x = F.relu(self.fc3(x))  
#         #x = self.dropout(x)
#         #x = F.relu(self.fc4(x))
        
#         # final output
#         return x
# ======================
class EMNISTCNN(nn.Module):
    '''
    Arguments:
    in_channels  – Number of channels in the input image.
    out_channels – Number of channels produced by the convolution.
    kernel_size – Size of the convolving kernel.
    stride – Stride of the convolution
    padding – Padding added to all four sides of the input.
    '''
    

    def __init__(self, fmaps1, fmaps2, dense, dropout):
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=fmaps1, kernel_size=5, stride=1, padding='same'),                              
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=fmaps1, out_channels=fmaps2, kernel_size=5, stride=1, padding='same'),                              
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fcon1 = nn.Sequential(nn.Linear(49*fmaps2, dense), nn.LeakyReLU())
        self.fcon2 = nn.Linear(dense, 47)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fcon1(x))
        x = self.fcon2(x)
        return x

# class EMNISTCNN(nn.Module):

    # def __init__(self):
    #     super(EMNISTCNN, self).__init__()
        
    #     # 1 input image channel (grayscale), 10 output channels/feature maps
    #     # 3x3 square convolution kernel
    #     self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    #     self.pool = nn.MaxPool2d(2,2)
    #     self.conv2 = nn.Conv2d(64,40,3, stride=1, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.fc1 = nn.Linear(7*7*40,256)
    #     self.dropout = nn.Dropout(0.4)
    #     self.fc2 = nn.Linear(256,47)
    #     self.fc3 = nn.Linear(47,64)
    #     self.fc4 = nn.Linear(64,47)
        
    #     ## TODO: Define the rest of the layers:
    #     # include another conv layer, maxpooling layers, and linear layers
    #     # also consider adding a dropout layer to avoid overfitting


    # ## TODO: define the feedforward behavior
    # def forward(self, x):
    #     # one activated conv layer
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = F.relu(self.fc2(x))
    #     x = self.dropout(x)
    #     x = F.relu(self.fc3(x))  
    #     x = self.dropout(x)
    #     x = F.relu(self.fc4(x))
        
    #     # final output
    #     return x