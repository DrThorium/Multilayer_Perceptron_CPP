#include "MultilayerPerceptron.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<std::string> split(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream(str);
    while (std::getline(token_stream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<double> one_at_index(int i, int features) {
    std::vector<double> v(features, 0.0);
    v[i] = 1.0;
    return v;
}


std::pair<int, int> processCSV(const std::string& filename, std::vector<std::pair<std::vector<double>, std::vector<double>>>& data, int features) {
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
        output = one_at_index(int(output[0]), features);
        data.emplace_back(input, output);
    }
    std::pair<int, int> sizes;
    sizes.first = data.front().first.size();
    sizes.second = data.front().second.size();
    return sizes;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        std::cout << "Error: Incorrect number of arguments." << std::endl;
        std::cout << "Usage: ./mlp {topology} learning_rate file_name #features" << std::endl;
        return 1;
    }

    std::vector<int> input_vector;
    std::string input_string = argv[1];
    input_string.erase(0, 1);
    input_string.erase(input_string.length() - 1, 1);
    std::vector<std::string> input_values = split(input_string, ',');
    for (const std::string& value : input_values) {
        input_vector.push_back(std::stoi(value));
    }

    double learning_rate = std::stod(argv[2]);

    std::string file_name = argv[3];

    std::vector<std::pair<std::vector<double>, std::vector<double>>> input_data;
    std::pair<int, int> io_size = processCSV(file_name, input_data, std::stoi(argv[4]));
    input_vector.push_back(io_size.second);
    input_vector.insert(input_vector.begin(), io_size.first);
    MLP::MultilayerPerceptron mlp_(input_vector);
    mlp_.trainMLP(input_data, learning_rate);

    return 0;
}