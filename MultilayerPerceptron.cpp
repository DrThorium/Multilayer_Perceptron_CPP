//
// Created by Juan Salazar on 30/12/22.
//

#include "MultilayerPerceptron.h"
#include <iostream>
#include <utility>
#include <random>

auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
auto sigmoidDer = [](double x) { auto s = sigmoid(x); return s * (1.0 - s); };
Eigen::VectorXd cross_entropy(const Eigen::VectorXd & actual, const Eigen::VectorXd & generated) {
    Eigen::VectorXd out_;
    out_.resize(actual.size());
    for (int i = 0; i < actual.size(); ++i) {
        out_[i] = actual[i] * std::log(generated[i]) + (1.0 - actual[i]) * std::log(1.0 - generated[i]);
    }
    return out_;
}

MLP::Layer::Layer(int size, int prev_size) {
    activations_.resize(size);
    biases_.resize(size);
    z_vectors_.resize(size);
    weights_.resize(prev_size, size);
}

MLP::Layer::Layer() {
    activations_.resize(1);
    biases_.resize(1);
    weights_.resize(1, 1);
    z_vectors_.resize(1);
}

void MLP::Layer::resizeLayer(int size, int prev_size, bool output_ = false) {
    activations_.resize(size);
    weights_.resize(prev_size, size);
    if (output_) return;
    else{
        biases_.resize(size);
        z_vectors_.resize(size);
    }
}

MLP::MultilayerPerceptron::MultilayerPerceptron(std::vector<int> topology)
        : topology_(std::move(topology)) {
    if (topology_.size() < 3) throw std::runtime_error("The Multilayer Perceptron needs at least 1 input "
                                                       "layer, 1 hidden layer and 1 output layer");
    output_layer_.resizeLayer(topology_.back(), topology_[topology_.size()-2], true);
    inputs_.resize(topology_.front());
    hidden_layers_.resize(topology_.size()-2);
    for (int i = 0; i < hidden_layers_.size(); i++){
        hidden_layers_[i].resizeLayer(topology_[i+1], topology_[i]);
    }
}

void MLP::MultilayerPerceptron::setRandom() {
    for (auto & hidden_layer : hidden_layers_){
        hidden_layer.biases_.setRandom();
        hidden_layer.weights_.setRandom();
    }
    output_layer_.weights_.setRandom();
}

void MLP::MultilayerPerceptron::forwardPropagation(const std::vector<double> &input) {
    if (input.size() != inputs_.size()) throw std::runtime_error("The size of the input differs from the Topology...");
    std::copy(input.begin(), input.end(), inputs_.data());
    hidden_layers_[0].z_vectors_ = (hidden_layers_[0].weights_.transpose() * inputs_) + hidden_layers_[0].biases_;
    hidden_layers_[0].activations_ = hidden_layers_[0].activations_.unaryExpr(sigmoid);
    for (int i = 1; i < hidden_layers_.size(); i++) {
        hidden_layers_[i].z_vectors_ = (hidden_layers_[i].weights_.transpose() * hidden_layers_[i-1].activations_)
                + hidden_layers_[i].biases_;
        hidden_layers_[i].activations_ = hidden_layers_[i].z_vectors_.unaryExpr(sigmoid);
    }
    output_layer_.z_vectors_ = hidden_layers_[hidden_layers_.size()-1].activations_.transpose() * output_layer_.weights_;
    output_layer_.activations_ = output_layer_.z_vectors_.unaryExpr(sigmoid);
}

void MLP::MultilayerPerceptron::trainMLP(std::vector<std::pair<std::vector<double>, std::vector<double>>> train_, double learning_rate) {
    learning_rate_ = learning_rate;
    setRandom();
    for (int i = 0; i < train_.size(); i++) {
        Eigen::VectorXd actual_output;
        actual_output.resize(train_[i].second.size());
        std::copy(train_[i].second.begin(), train_[i].second.end(), actual_output.data());
        std::cout << "Train #" << i << "." << std::endl;
        forwardPropagation(train_[i].first);
        std::cout << "Expected Output: " << actual_output << std::endl;
        std::cout << "Output generated: " << output_layer_.activations_ << std::endl;
        std::cout << "Cross Entropy:" << cross_entropy(actual_output, output_layer_.activations_).minCoeff() << std::endl;
        backwardPropagation(train_[i].second);
    }
}

void MLP::MultilayerPerceptron::backwardPropagation(const std::vector<double> &output) {
    calculateDeltas(output);
    updateWeights();
    updateBiases();
}

void MLP::MultilayerPerceptron::calculateDeltas(const std::vector<double> &output) {
    // Prepare the output VectorXd.
    Eigen::VectorXd actual_output;
    actual_output.resize(output.size());
    std::copy(output.begin(), output.end(), actual_output.data());
    // Calculate the Delta of the output layer and for the last hidden layer.
    output_layer_.Delta_ = output_layer_.z_vectors_.unaryExpr(sigmoidDer).cwiseProduct(2*(actual_output-output_layer_.activations_) );
    Eigen::MatrixXd temp = output_layer_.weights_ * output_layer_.Delta_; // for reliability.
    hidden_layers_[hidden_layers_.size()-1].Delta_ = hidden_layers_[hidden_layers_.size()-1].z_vectors_.unaryExpr(sigmoidDer).cwiseProduct(temp);
    // Calculate the Delta for all the Hidden Layers.
    for (int i = hidden_layers_.size()-2; i >= 0; i--) {
        temp = hidden_layers_[i+1].weights_ * hidden_layers_[i+1].Delta_;
        hidden_layers_[i].Delta_ = hidden_layers_[i].z_vectors_.unaryExpr(sigmoidDer).cwiseProduct(temp);
    }
}

void MLP::MultilayerPerceptron::updateWeights() {
    Eigen::MatrixXd temp;
    //for the output
    temp = learning_rate_ * (output_layer_.Delta_ * hidden_layers_[hidden_layers_.size()-1].activations_.transpose());
    output_layer_.weights_ = output_layer_.weights_ + temp.transpose();
    //for the Hidden Layers
    if (hidden_layers_.size() > 1) for (int i = 1; i < hidden_layers_.size()-1; i++) {
            temp = learning_rate_ * (hidden_layers_[i].Delta_ * hidden_layers_[i-1].activations_.transpose());
            hidden_layers_[i].weights_ = hidden_layers_[i].weights_ + temp.transpose();
        }
    //for the first Hidden Layer (which needs the input):
    temp = learning_rate_ * (hidden_layers_[0].Delta_ * inputs_.transpose());
    hidden_layers_[0].weights_ = hidden_layers_[0].weights_ + temp.transpose();
}

void MLP::MultilayerPerceptron::updateBiases() {
    Eigen::VectorXd temp;
    if (hidden_layers_.size() > 1)
        for (int i = 1; i < hidden_layers_.size() - 1; i++) {
            temp = learning_rate_ * hidden_layers_[i].Delta_;
            hidden_layers_[i].biases_ = hidden_layers_[i].biases_ + temp;
        }
    temp = learning_rate_ * hidden_layers_[0].Delta_;
    hidden_layers_[0].biases_ = hidden_layers_[0].biases_ + temp;
}


