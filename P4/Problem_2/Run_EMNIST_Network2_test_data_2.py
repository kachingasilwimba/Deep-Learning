# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 19:00:48 2022

@author: johnchiasson
"""
import json  #Java Script Object Notation
import sys  #System-specific parameters and functions module
import numpy as np
import matplotlib.pyplot as plt
import EMNIST_Loader2
import network2_EMNIST

f = open('EMNIST_plot_data.json', "r")
[validation_cost, validation_accuracy, training_cost, training_accuracy] = json.load(f)
f.close()
f = open('EMNIST_plot_parameters.json', "r")
[num_epochs, training_cost_xmin, validation_accuracy_xmin, validation_cost_xmin, 
               training_accuracy_xmin, training_set_size,validation_set_size] = json.load(f)
f.close()

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
            [accuracy/100.0/validation_set_size
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
            [accuracy*100.0/training_set_size 
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
            [accuracy*100.0/validation_set_size for accuracy in validation_accuracy], 
            color='#2A6EA6',
            label="Accuracy on the validation data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
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
