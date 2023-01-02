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

Notice that I only make a "Single output" MLP, but it can handle std::vector<double> of any size, if you need it. As you can notice, there's still no "Test" function, so neither the Epoch Hyperparameter, the MLP generates a Cross Entropy Error for each training (observation in the data). Of course, after all trainings, you can simply make a Feed Forward Propagation with some of your Test Inputs.

./mlp {TOPOLOGY} INT_LEARNING_RATE STRING_FileName int_ClassificationPossibleValues

Here's an example:

./mlp 16,16,16,23,42 0.998421 Red_wine.csv 10

This example executes a training along all the red_wine.csv with a 0.998421 learning rate, with 5 hidden layers of Neuron Sizes of 16,16,16,23,42 and prepares an output of classification for 10 possible values.
(Note: The MLP is fitted for classification only, but can do a regression with values in the range of 0.0-1.0, which is the activation value of the output neurons).

 <img width="270" alt="image" src="https://user-images.githubusercontent.com/119984041/210188475-c2a3a22b-3a52-4ac6-8aac-2e94ddf033bb.png">
