import sys
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
#pip install torchsummary
from torchsummary import summary
#pip install extra-keras-datasets
from extra_keras_datasets import emnist

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device '.format(device)) 


#========================Load EMNIST balanced testdataset===============================

(input_train, target_train), (input_test, target_test) = emnist.load_data(type='balanced')
test_images = torch.tensor((input_test).reshape(18800, 1, 28, 28))
test_set = list(zip(test_images.float(), target_test.astype('int64')))



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
batch_size = 20
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
print()
cnn_model = torch.load('EMNIST_CNN_trained.json')
cnn_model.to(device)
test(cnn_model)

print()
print('============= EMNIST Test Data Shape and Type ===============')
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
num_epochs = 20
f = open('EMNIST_plot_data.json', "r")
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
    plot_overlay(validation_accuracy, training_accuracy, num_epochs,
                 min(validation_accuracy_xmin, training_accuracy_xmin),
                 training_set_size)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_validation_accuracy(validation_accuracy, num_epochs, validation_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_accuracy_xmin, num_epochs), 
            [accuracy
             for accuracy in validation_accuracy[validation_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([validation_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the validation data')
    plt.show()

def plot_validation_cost(validation_cost, num_epochs, validation_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(validation_cost_xmin, num_epochs), 
            validation_cost[validation_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([validation_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the validation data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

def plot_overlay(validation_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy for accuracy in validation_accuracy], 
            color='#2A6EA6',
            label="Accuracy on the validation data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy
             for accuracy in training_accuracy], 
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([0, 100])
    plt.legend(loc="lower right")
    plt.show()

make_plots('EMNIST_plot_data.json', num_epochs, training_cost_xmin, validation_accuracy_xmin,
           validation_cost_xmin,  training_accuracy_xmin, training_set_size)
