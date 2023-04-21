import sys
import torch
from torch import nn
import FashionMNIST_CNN_Network
import math
import torch
import torchvision
from torchvision import datasets, transforms
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import json
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device '.format(device))


#=======================Load Fashion MNIST Dataset================================
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

#==========================Download and load the training data
train_set = datasets.FashionMNIST(root='./data/FashionMNIST', download = True, train = True, transform = transform)
test_set = datasets.FashionMNIST(root='./data/FashionMNIST', download = True, train = False, transform = transform)

label_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

#===========================split training data into training subset and validation subset
idxs = torch.randperm(len(train_set)).tolist()

FMNIST_train_set = torch.utils.data.Subset(train_set, idxs[:50000])
FMNIST_val_set = torch.utils.data.Subset(train_set, idxs[50000:])

#==========================Fashion MNIST Data Description============================
print()
print('=============Fashion MNIST Training Data Shape and Type ========================')
print('type(training_data) =', type(FMNIST_train_set))
print('len(training_data) =', len(FMNIST_train_set))
print('type(training_data[0]) =', type(FMNIST_train_set[0]))
print('type(training_data[1]) =', type(FMNIST_train_set[1]))
print('len(training_data[0]) =', len(FMNIST_train_set[0]))
print('training_data[0][0].shape = ', FMNIST_train_set[0][0].shape)
print('training_data[0][1] =', FMNIST_train_set[0][1])
img,label = FMNIST_train_set[0]
print('img.shape =', img.shape)
print('label =',label)

print()
print('=============CIFAR100 Validation Data Shape and Type ====================')
print('type(validation_data) =', type(FMNIST_val_set))
print('len(validation_data) =', len(FMNIST_val_set))
print('type(validation_data[0]) =', type(FMNIST_val_set[0]))
print('type(validation_data[1]) =', type(FMNIST_val_set[1]))
print('len(validation_data[0]) =', len(FMNIST_val_set[0]))
print('validation_data[0][0].shape = ', FMNIST_val_set[0][0].shape)
print('validation_data[0][1] =', FMNIST_val_set[0][1])
img,label = FMNIST_val_set[0]
print('img.shape =', img.shape)
print('label =',label) 
print()

#====================================================================================
#                    Tranining Convolution Neural network
#====================================================================================

def train_loop(cnn_model, optimizer, loss_fn, batch_size):
    '''
    Returns validation loss and accuracy
        Parameters:
            cnn_model (CNN): a convolutional neural network to train
            optimizer: optimizer
             loss function: a loss function to evaluate the model on 
        Returns:
            cnn_model (CNN): a trained model
            train_loss (float): train loss
            train_acc (float): train accuracy
    '''
    train_loader = torch.utils.data.DataLoader(FMNIST_train_set, batch_size = batch_size, shuffle = True)
    cnn_model.train()
    correct = 0
    total = 0
    train_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = cnn_model(inputs)
        
        optimizer.zero_grad()
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        #===================== the class with the highest value is the prediction
        _, prediction = torch.max(outputs.data, 1)  #====== grab prediction as one-dimensional tensor
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    training_loss = train_loss/len(train_loader)
    training_accs = 100*correct/total
    return cnn_model, training_loss, training_accs  

#====================================================================================
#                    Validation of Convolution Neural network
#====================================================================================

def valid_loop(cnn_model, loss_fn ,batch_size):
    '''
    Returns validation loss and accuracy
    
        Parameters:
            cnn_model (CNN): a convolutional neural network to validate
            loss function: a loss function to evaluate the model on
        
        Returns:
            validation_loss (float): validation loss
            validation_accs (float): validation accuracy
    '''
    
    val_loader =  torch.utils.data.DataLoader(FMNIST_val_set, batch_size = batch_size, shuffle = True)    
    cnn_model.eval()
    correct = 0
    total = 0
    val_loss = 0 
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = cnn_model(inputs)

            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
        validation_loss = val_loss/len(val_loader)   
        validation_accs = 100*correct/total

    return validation_accs, validation_loss


#====================================================================================
#                    Tranining and Save Convolutional Neural network
#====================================================================================

def train_save():   
    '''
    Execute train and validate functions epoch-times to train a CNN model.
    Each time, store train & validation loss and accuracy.
    Then, test the model and return the result.
    '''
    cnn_model =  FashionMNIST_CNN_Network.Fashion_MNIST_Network().to(device)

    loss_fn = nn.CrossEntropyLoss()
    #=====================optimizer
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr = 0.0001) 
    #optimizer = torch.optim.SGD(cnn_model.parameters(), lr = 0.05, weight_decay = 0.0001)# learning rate
    
    num_epochs = 30
    batch_size = 90
    
    #=====================containers to keep track of statistics
    training_cost = []
    validation_cost = []
    training_accuracy = []
    validation_accuracy  = []
       
    for epoch in range(num_epochs):  #=====================number of training to be completed
        cnn_model, training_loss, training_accs = train_loop(cnn_model, optimizer, loss_fn, batch_size)
        validation_accs, validation_loss = valid_loop(cnn_model, loss_fn, batch_size)
        
        training_cost.append(training_loss)
        validation_cost.append(validation_loss)
        training_accuracy.append(training_accs)
        validation_accuracy.append(validation_accs)
        
    #=====================print results of each iteration
        print(f'Epoch [{epoch+1}/{num_epochs}]=======================================================================')
        print(f'Accuracy(train, validation):{round(training_accs,1),round(validation_accs,1)}%, Loss(train,validation):{round(training_loss,4), round(validation_loss,4)}')
        print()
    torch.save(cnn_model, 'FASHIONMNIST_CNN_trained.json') 
    filename = 'FASHIONMNIST_plot_data.json'
    f = open(filename, "w")
    json.dump([validation_cost, validation_accuracy, training_cost, training_accuracy], f)
    f.close() 
train_save()
