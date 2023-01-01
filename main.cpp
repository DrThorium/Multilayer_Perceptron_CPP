#include "MultilayerPerceptron.h"
#include <fstream>
#include <sstream>

std::vector<double> one_at_index(int i) {
    std::vector<double> v(10, 0.0);
    v[i] = 1.0;
    return v;
}


void processCSV(const std::string& filename, std::vector<std::pair<std::vector<double>, std::vector<double>>>& data) {
    std::ifstream file(filename);

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        std::vector<double> input;
        std::vector<double> output;

        while (std::getline(lineStream, cell, ',')) {
            input.push_back(std::stod(cell));
        }

        output.push_back(input.back());
        input.pop_back();
        output = one_at_index(int(output[0]));
        data.emplace_back(input, output);
    }
}


int main() {
    MLP::MultilayerPerceptron mlp_({11,16,16,16,10});
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data;
    processCSV("winequality-red.csv", data);
    mlp_.trainMLP(data);
}
