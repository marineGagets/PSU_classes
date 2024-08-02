# Programmin Assignment - HW1 -CS445
# Larry Bilodeau - July 2024

# Implements a 2 layer neural node machine learning algorithm to classify
#  tne set of data points given in https/www.kaggle.com/oddratrionale/mnist-in-cvs\
#  python3 programming%20assignment%201-LBilodeau.py hidden_nodes eta momentum output_nodes max_epochs  

# usage:
#  There are default values set for each of these parameters or you can specify them on the command line
#   arg 1: Number of hidden node, hidden_nodes, default = 20
#   arg 2: Learning rate, eta, default = 0.1
#   arg 3: Momentum, momentum, default = 0.9
#   arg 4: Number of output nodes, output_nodes, default = 10
#   arg 5: Number of input nodes, input_nodes, default = 784
#   arg 6: Maximum number of epochs, max_epochs, default = 50


# Load python packages for the operating system interactions, the matrix math, and the plotting packages\
import os
import sys
import numpy as np
import pylab as pl
import os
import pandas as pd   #use this package to load data from the internet (url)


global targets, inputs, hidden_weights, hidden_bias, output_weights, output_bias, change_hidden_weights, \
           change_hidden_bias, change_output_weights, change_output_bias, eta, momentum, max_epochs, \
           rawdata, hidden_nodes, output_nodes, input_nodes, output_bias_weights, hidden_bias_weights

# define the main __init__  function


import pandas as pd
import numpy as np
import sys

def initilize_global():
    """
    Initialize global variables and load the MNIST training dataset.

    Returns:
        targets (pd.Series): The target values.
        inputs (pd.DataFrame): The preprocessed input values.
        hidden_weights (np.ndarray): The weights for the hidden layer.
        hidden_bias (np.ndarray): The biases for the hidden layer.
        output_weights (np.ndarray): The weights for the output layer.
        output_bias (np.ndarray): The biases for the output layer.
        change_hidden_weights (np.ndarray): The change in weights for the hidden layer.
        change_hidden_bias (np.ndarray): The change in biases for the hidden layer.
        change_output_weights (np.ndarray): The change in weights for the output layer.
        change_output_bias (np.ndarray): The change in biases for the output layer.
        eta (float): The learning rate.
        momentum (float): The momentum.
        max_epochs (int): The maximum number of epochs.
        rawdata (pd.DataFrame): The raw data.
        hidden_nodes (int): The number of nodes in the hidden layer.
        output_nodes (int): The number of nodes in the output layer.
        input_nodes (int): The number of nodes in the input layer.
        output_bias_weights (np.ndarray): The weights for the output bias.
        hidden_bias_weights (np.ndarray): The weights for the hidden bias.
    """
    # Load the mnist training dataset from the local file for the assignment into a 2 dimensional array
    rawdata = pd.read_csv('D:\mnist_train.csv')  # load the data from the local file
    print("raw_Data shape: ")
    print(np.shape(rawdata))   # verify the array's size (shape).

    # Set the machine's learning parameters to the values provided in the command line if present else set them to a default value.
    # - number of hidden layer nodes, hidden(n)
    if len(sys.argv) > 1:
        hidden_nodes = int(sys.argv[1])
    else:
        hidden_nodes = 20  # default value if no command line parameter is provided
    # - learning rate, eta
    if len(sys.argv) > 2:
        eta = float(sys.argv[2])
    else:
        eta = 0.1  # default value if no command line parameter is provided
    # - momentum
    if len(sys.argv) > 3:
        momentum = float(sys.argv[3])
    else:
        momentum = 0.9  # default value if no command line parameter is provided
    # - number of output nodes, output(k)
    if len(sys.argv) > 4:
        output_nodes = int(sys.argv[4]) 
    else:
        output_nodes = 10
    # - number of input nodes, input(i)
    if len(sys.argv) > 5:
        input_nodes = int(sys.argv[5])
    else:
        input_nodes = 784  # 28 x 28 pixel images
    # - maximum number of epochs     
    if len(sys.argv) > 6:
        max_epochs = int(sys.argv[6])   
    else:
        max_epochs = 50

    # Trim inputs randomly to max_epochs length       
    rawdata = rawdata.sample(n=max_epochs, random_state=42)
    # Split the preprocess data into inputs and targets 
    inputs = rawdata.iloc[:, 0:]  # all columns except the first
    # Normalize the input data
    inputs = inputs / 255
    print("first row the inputs sets")    
    print(inputs.iloc[0, :].values)

    targets = rawdata.iloc[:, 0]  # the first column
    print("First target values:")
    print(targets.iloc[0])
    # Print out the tables of inputs, activations, weights, and errors for the hidden and output nodes for the epochs
    # Print out raw data and the corresponding preprocessed inputs
    # Insert column headers as the first row for the 2 arrays
    #column_headers = pd.DataFrame({"Target": [], "Raw Input Values": []}, index=[0])
    #rawdata = pd.concat([column_headers, rawdata[:]]).reset_index(drop=True) 
    #column_headers = pd.DataFrame({"Preprocessed Input Values": [], "Raw Input Values": []}, index=[1])
    #tmp_inputs = pd.concat([column_headers, inputs[:]]).reset_index(drop=True)    
    


    # Initialize the weights to random values (-0.5 < w < 0.5) and the biases to 1.0
    hidden_weights = np.random.rand(input_nodes, hidden_nodes) - 0.5
    hidden_bias = np.ones(hidden_nodes)  # set the bias to 1.0
   
    output_weights = np.random.rand(hidden_nodes,output_nodes) - 0.5 # set the bias to random values beteen -0.5 and 0.5
    output_bias = np.ones(output_nodes)
    
    # Initialize the change in weights and biases to zero
    change_hidden_weights = np.zeros((hidden_nodes, input_nodes))
    change_hidden_bias = np.zeros((input_nodes, hidden_nodes))
    change_output_weights = np.zeros((output_nodes, hidden_nodes))
    change_output_bias = np.zeros((output_nodes, 1))

    return targets, inputs, hidden_weights, hidden_bias, output_weights, output_bias, change_hidden_weights, \
           change_hidden_bias, change_output_weights, change_output_bias, eta, momentum, max_epochs, \
           rawdata, hidden_nodes, output_nodes, input_nodes, output_bias



# define the confusion matrix function
def confusion_matrix(inputs, targets, hidden_weights, hidden_bias, output_weights, output_bias):
    # forward pass
    hidden_activations = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_activations = 1 / (1 + np.exp(-hidden_activations))
    output_activations = np.dot(hidden_activations, output_weights) + output_bias
    output_activations = 1 / (1 + np.exp(-output_activations))

    # convert the output activations to a one-hot encoding
    outputs = np.zeros_like(output_activations)
    outputs[np.arange(len(output_activations)), output_activations.argmax(1)] = 1

    # convert the targets to a one-hot encoding
    targets_one_hot = np.zeros_like(outputs)
    targets_one_hot[np.arange(len(targets)), targets] = 1

    # calculate the confusion matrix
    confusion_matrix = np.dot(targets_one_hot.T, outputs)

    return confusion_matrix

# define the sigmoid function and it's derivative
def sigmoid(x):
    if (1 / (1 + np.exp(-x)) >= 0.9):
        return 0.9
    else: return 0.1

def sigmoid_derivative(x):
    return x * (1 - x)


# the training function

def train(hidden_weights, hidden_bias, output_weights, output_bias, change_hidden_weights,
            change_hidden_bias, change_output_weights, change_output_bias, eta, momentum, max_epochs \
            ):

    # define an array to store hidden nodes values, weights, and error values for each epoch
    # and an array for the output nodes values, weights, and error values for each epoch
    # and an array for the hidden and outout biases for each epoch

    hidden_nodes_activation = np.zeros((max_epochs, hidden_nodes))
    hidden_weight_values = np.zeros((max_epochs, hidden_weights.shape[0],hidden_weights.shape[1]))
    hidden_error_values = np.zeros((max_epochs, hidden_weights.shape[0], hidden_weights.shape[1]))
    output_nodes_activation = np.zeros((max_epochs, output_nodes))
    output_weight_values = np.zeros((max_epochs, output_nodes, hidden_nodes))
    output_errors_values= np.zeros((max_epochs, output_nodes))
    output_bias_values = np.ones((max_epochs, output_nodes))
    output_bias_weights = np.random.rand(output_nodes, hidden_nodes) - 0.5  # set the bias weights to random values beteen -0.5 and 0.5
    hidden_bias_weights = np.random.rand(hidden_nodes, input_nodes) - 0.5  # set the bias weights to random values beteen -0.5 and 0.5  
    hidden_bias_values = np.ones((max_epochs, hidden_nodes))
    output_deltas = np.zeros((max_epochs, output_nodes))
    hidden_deltas = np.zeros((max_epochs, hidden_nodes))

    for epoch in range(max_epochs):
        print("epoch: ", epoch)

        # forward pass
        hidden_activations = np.dot(inputs.iloc[epoch, :], hidden_weights.T) + hidden_bias
        print("hidden_activations")
        print(hidden_activations)
        
        hidden_activations = 1 / (1 + np.exp(-hidden_activations))
        print("updated hidden_activations")
        hidden_nodes_activation[epoch,:] = hidden_activations
        print(hidden_activations)
        print("pre calculation output_activations")
        #print(output_activations)
        output_activations = np.dot(hidden_activations, output_weights) + output_bias
        output_activations = 1 / (1 + np.exp(-output_activations))
        print("output_activations")  
        print(output_activations)
        output_nodes_activation[epoch,:] = output_activations

        # backward pass
        # reinturpt the target values as a one-hot encoding
        targets_one_hot = np.zeros(output_nodes)
        targets_one_hot[np.arange(output_nodes)] = 0.1
        targets_one_hot[int(targets.iloc[epoch])] = 0.9        
        print("targets_one_hot, shape: ", np.shape(targets), "target: ", targets.iloc[epoch])
        print(targets_one_hot)

        # calculate the errors and deltas
        output_error = targets_one_hot - output_activations
        output_errors_values[epoch, :] = output_error
        print("output_error")
        print(output_error)
        output_delta = output_error * sigmoid_derivative(output_activations)
        print("output deltas:", output_delta)
        output_deltas[epoch, :] = output_delta
        hidden_error = output_delta.dot(output_weights.T)
        hidden_error_values[epoch,:] = hidden_error
        print("hidden_error")
        print(hidden_error)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_activations)
        print("hidden_deltas:", hidden_delta)
        hidden_deltas[epoch, :] = hidden_delta

        # update weights and biases
        # Add the bias values to the weights
        if epoch == 0:
            change_output_weights = eta * hidden_activations
            change_hidden_weights = eta * inputs.iloc[epoch, :]
            change_output_bias = eta * np.dot(hidden_activations, output_bias_weights.T)
            change_hidden_bias = eta * np.dot(inputs.iloc[epoch, :], hidden_bias_weights.T) 
        else: 
            change_output_bias = eta * output_delta.sum(axis=0)
            change_hidden_weights = eta * inputs.iloc[epoch, :].T * hidden_delta +  \
                momentum * change_hidden_weights
            change_hidden_bias = eta * hidden_delta.sum(axis=0)
            change_output_weights = eta * hidden_activations * output_delta + momentum * change_output_weights
            change_output_bias = eta * output_delta.sum(axis=0) + momentum * change_output_bias
        output_weights += np.expand_dims(change_output_weights, 1)
        output_weight_values = output_weights.T

        #output_bias += np.expand_dims(change_output_bias,1)
        output_bias += change_output_bias
        print("back output_bias")
        print(output_bias)
        output_bias_values = output_bias.T

        hidden_weights += np.expand_dims(change_hidden_weights, 1)
        hidden_weight_values = hidden_weights.T

        hidden_bias += change_hidden_bias
        hidden_bias_values = hidden_bias.T

        print("for epoch: ", epoch)
        print()
        print("OUTPUT LAYER:")
        print("output activation, shape: ", np.shape(output_activations))    
        print(output_activations)
        print("output weights, shape:", np.shape(output_weights))
        print(output_weights)
        print("output bias, shape:", np.shape(output_bias))
        print(output_bias)
        print("output error, shape:", np.shape(output_error))
        print(output_error)
        print("output deltas, sahpe:", np.shape(output_delta))
        print(output_delta)
        print()
        print("HIDDEN LAYER:")
        print("output activation shape: ", np.shape(hidden_activations))
        print(hidden_activations)
        print("output weights, shape:", np.shape(hidden_weights))
        print(hidden_activations)
        print("hidden weights shape:", np.shape(hidden_weights))
        print(hidden_weights)
        print("hidden bias shape:", np.shape(hidden_bias))
        print(hidden_bias)
        print("hidden error shape:", np.shape(hidden_error))
        print(hidden_error)
        print("hidden deltas shape:", np.shape(hidden_delta))
        print(hidden_delta)

    return targets, inputs, hidden_weights, hidden_bias, output_weights, output_bias, change_hidden_weights, \
        change_hidden_bias, change_output_weights, change_output_bias, output_nodes_activation, \
        output_weight_values, output_errors_values, output_bias_values, hidden_nodes_activation, \
        hidden_weight_values, hidden_error_values, hidden_bias_values

targets, inputs, hidden_weights, hidden_bias, output_weights, output_bias, change_hidden_weights, \
           change_hidden_bias, change_output_weights, change_output_bias, eta, momentum, max_epochs, \
           rawdata, hidden_nodes, output_nodes, input_nodes, output_bias=initilize_global()

# train the neural network

targets, inputs, hidden_weights, hidden_bias, output_weights, output_bias,\
    change_hidden_weights, change_hidden_bias, change_output_weights, change_output_bias,\
    output_nodes_activation, output_weight_values, output_errors_values, output_bias_values,\
    hidden_nodes_activation, hidden_weight_values, hidden_error_values, \
        hidden_bias_values = train(hidden_weights, \
        hidden_bias, output_weights, output_bias, change_hidden_weights, change_hidden_bias, \
        change_output_weights, change_output_bias, eta, momentum, max_epochs)

# print out the hidden nodes activation, weight values, error values, and bias values for each epoch
print()
print("Hidden Nodes Activation")
print(hidden_nodes_activation)
print()
print("Hidden Weight Values")   
print(hidden_weight_values)
print()
print("Hidden Error Values")
print(hidden_error_values)
print()
print()
print("Hidden Bias Values")
print(hidden_bias_values)
# print out the output nodes activation, weight values, error values, and bias values for each epoch
print()
print("Output Nodes Activation")
print(output_nodes_activation)
print()
print("Output Weight Values")
print(output_weight_values)
print()
print("Output Errors Values")
print(output_errors_values)
print()
print("Output Bias Values")
print(output_bias_values)





#calculate the confusion matrix
confusion_matrix = confusion_matrix(input, targets, hidden_weights, hidden_bias, output_weights, output_bias)    
print()
print("Confusion Matrix")
