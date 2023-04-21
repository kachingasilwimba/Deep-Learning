# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:00:48 2022

@author: johnchiasson
"""
import json  #Java Script Object Notation
import sys  #System-specific parameters and functions module
import numpy as np
import MNIST_Loader
import network2

training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()

net = network2.load('MNIST_CrossEntropy.json')
print()
print('net.weights[0].shape =', net.weights[0].shape)
print('net.weights[1].shape =', net.weights[1].shape)
print('net.biases[0].shape =', net.biases[0].shape)
print('net.biases[1].shape =', net.biases[1].shape)
print()
print('net.cost =', net.cost)
print()
print('accuracy on test data =', net.accuracy(test_data, convert=False))
