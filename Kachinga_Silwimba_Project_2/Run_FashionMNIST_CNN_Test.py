import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device '.format(device)) 


#========================Load Fashion MNIST dataset===============================

#========================Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
#========================Download and load the training data
test_set = datasets.FashionMNIST(root='./data/FashionMNIST', download = True, train = False, transform = transform)
train_set = datasets.FashionMNIST(root='./data/FashionMNIST', download = True, train = True, transform = transform)
label_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


#====================================================================================
#                    Test Convolution Neural network Saved
#====================================================================================

def test(cnn_model):
    '''Calculates the accuracy of the CNN on the test data'''
    size = len(test_loader.dataset)
    cnn_model.eval()
    with torch.no_grad():
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_output = cnn_model.forward(images)
            pred_y = torch.max(test_output, 1)[1]
            correct += (pred_y == labels).sum()
    accuracy = (correct*100/size) #========Our test data has 18800 images
    print('Test Data Accuracy: {0:.2f}'.format(accuracy))
    return accuracy

#================================Load and test CNN=====================================
if __name__ == '__main__':
    batch_size = 300
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = True)
    print()
    cnn_model = torch.load('FASHIONMNIST_CNN_trained.json')
    cnn_model.to(device)
    test(cnn_model)

    print()
    print('=============Fashion MNIST Test Data Shape and Type ===============')
    print('type(test_data) =', type(test_set))
    print('len(test_data) =', len(test_set))
    print('type(test_data[0]) =', type(test_set[0]))
    print('type(test_data[1]) =', type(test_set[1]))
    print('len(test_data[0]) =', len(test_set[0]))
    print('test_data[0][0].shape = ', test_set[0][0].shape)
    print('test_data[0][1] =', test_set[0][1])
    img,label = test_set[0]
    print('img.shape =', img.shape)
    print('label =',label)
    print()
    #summary(cnn_model, (1, 28, 28))

#=========================================================================================
#     Plotting validation_cost, validation_accuracy, training_cost, training_accuracy
#=========================================================================================
num_epochs = 30
f = open('FASHIONMNIST_plot_data.json', "r")
[validation_cost, validation_accuracy, training_cost, training_accuracy] = json.load(f)
f.close()

training_cost_xmin = 0
validation_accuracy_xmin = 0
validation_accuracy_xmin = 0
validation_cost_xmin = 0
training_accuracy_xmin = 0
training_set_size = 100000
validation_set_size = 12800

def make_plots(filename, num_epochs, 
               training_cost_xmin, 
               validation_accuracy_xmin, 
               validation_cost_xmin, 
               training_accuracy_xmin,
               training_set_size):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    validation_cost, validation_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_validation_accuracy(validation_accuracy, num_epochs, validation_accuracy_xmin)
    plot_validation_cost(validation_cost, num_epochs, validation_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='r', linewidth =2)
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cost on the Fashion-MNIST Training Data',fontweight='bold')
    plt.show()

def plot_validation_accuracy(validation_accuracy, num_epochs, validation_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_accuracy_xmin, num_epochs), 
            [accuracy
             for accuracy in validation_accuracy[validation_accuracy_xmin:num_epochs]],
            color='g', linewidth =2)
    ax.set_xlim([validation_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy (%) on the Fashion-MNIST Validation Data',fontweight='bold')
    plt.show()

def plot_validation_cost(validation_cost, num_epochs, validation_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_cost_xmin, num_epochs), 
            validation_cost[validation_cost_xmin:num_epochs],
            color='g', linewidth =2)
    ax.set_xlim([validation_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Cost on the Fashion-MNIST Validation Data',fontweight='bold')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='r', linewidth =2)
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy (%) on the Fashion-MNIST Training Data',fontweight='bold')
    plt.show()

make_plots('FASHIONMNIST_plot_data.json', num_epochs, training_cost_xmin, validation_accuracy_xmin,
           validation_cost_xmin,  training_accuracy_xmin, training_set_size)
