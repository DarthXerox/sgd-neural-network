#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H

#include "NeuronLayer.h"
#include "InputManager.h"
#include <vector>

template<typename F = float>
struct NeuralNetwork {
    NeuralNetwork(const std::string& training_file, const std::string& training_labels);
    void start_testing(const std::string& test_file, const std::string& output_file);

private:
    TrainingInputIterator training_input_iterator;
    TestInputIterator test_input_iterator;
    std::vector<NeuronLayer<F>> layers;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
