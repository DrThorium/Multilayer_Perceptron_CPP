# Multilayer_Perceptron_CPP
An implementation of the Forward and Backward Propagation functions for a multi-layer perceptron with an input layer, N (greater than 0) hidden layers, and an output layer. All in C++ using Eigen for vector/matrix operations.

# Multi-Layer Perceptron in C++ using Eigen
This repository contains a C++ implementation of a multi-layer perceptron (MLP) using the Eigen library. The MLP has an input layer, N (greater than 0) hidden layers, and an output layer.

## Functions
The following functions are implemented:

- Forward Propagation: Propagates the input through the network and calculates the output of the MLP.

- Backward Propagation: Propagates the error back through the network and updates the weights and biases of the MLP.

- Train: Trains the MLP on a collection of inputs and expected outputs.

## Dependencies
This code depends on the Eigen library. Make sure that you have it installed on your system before attempting to compile the code.

## Compiling and Running
To compile the code, use the following command:

g++ -std=c++11 mlp.cpp -o mlp

## To run the code, use the following command:
(Notice that I only make a "Single output" MLP, but it can handle std::vector<double> of any size, if you need it).
As you can notice, there's still no "Test" function, so neither the Epoch Hyperparameter, the MLP generate a Cross Entropy Error for each train (Observation in the data).

./mlp {TOPOLOGY} INT_LEARNING_RATE STRING
