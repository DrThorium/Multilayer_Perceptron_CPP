//
// Created by Juan Salazar on 30/12/22.
//

#ifndef MULTILAYER_PERCEPTRON_MULTILAYERPERCEPTRON_H
#define MULTILAYER_PERCEPTRON_MULTILAYERPERCEPTRON_H

#include <Eigen/Eigen>
#include <vector>

namespace MLP{
    class MultilayerPerceptron;

    class Layer {
    protected:
        Eigen::VectorXd activations_;
        Eigen::VectorXd z_vectors_;
        Eigen::VectorXd biases_;
        Eigen::MatrixXd weights_;
        Eigen::VectorXd Delta_;
        Layer(int size, int prev_size);
        void resizeLayer(int size, int prev_size, bool output_);
        friend class MultilayerPerceptron;
    public:
        Layer();
    };
    class MultilayerPerceptron {
    private:
        double learning_rate_ = 1;
        std::vector<int> topology_;
        Eigen::VectorXd inputs_;
        std::vector<Layer> hidden_layers_;
        Layer output_layer_;
        void setRandom();
        void forwardPropagation(const std::vector<double>& input);
        void backwardPropagation(const std::vector<double>& output);
        void calculateDeltas(const std::vector<double> &output);
        void updateWeights();
        void updateBiases();
    public:
        explicit MultilayerPerceptron(std::vector<int> topology);
        void trainMLP(std::vector<std::pair<std::vector<double>, std::vector<double>>> train_);
    };
}

#endif //MULTILAYER_PERCEPTRON_MULTILAYERPERCEPTRON_H
