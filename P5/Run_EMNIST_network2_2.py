# Standard library
import json
import random
import sys
# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
# My library
import EMNIST_Loader2
import network2_EMNIST

training_data, validation_data, test_data = EMNIST_Loader2.load_data_wrapper()

net = network2_EMNIST.Network([784, 100, 47], cost=network2_EMNIST.CrossEntropyCost)

num_epochs = 30

validation_cost, validation_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data, num_epochs, 10, 0.5,
                  evaluation_data=validation_data, lmbda = 5,
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
 
net.save('EMNIST_CrossEntropy.json')
filename = 'EMNIST_plot_data.json'
f = open(filename, "w")
json.dump([validation_cost, validation_accuracy, training_cost, training_accuracy], f)
f.close() 

training_cost_xmin = 0
validation_accuracy_xmin = 0
validation_accuracy_xmin = 0
validation_cost_xmin = 0
training_accuracy_xmin = 0
training_set_size = len(training_data)
validation_set_size = len(validation_data)

filename = 'EMNIST_plot_parameters.json'
f = open(filename, "w")
json.dump([num_epochs, training_cost_xmin, validation_accuracy_xmin, validation_cost_xmin, 
               training_accuracy_xmin, training_set_size,validation_set_size], f)
f.close()

########################################################
print()
print('net.weights[0].shape =', net.weights[0].shape)
print('net.weights[1].shape =', net.weights[1].shape)
print('net.biases[0].shape =', net.biases[0].shape)
print('net.biases[1].shape =', net.biases[1].shape)
print()
print('net.cost =', net.cost)
print()
print('Accuracy on Test Data = ', net.accuracy(test_data, convert=False)/len(test_data))
print()