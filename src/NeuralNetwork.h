#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H

#include "NeuronLayer.h"
#include "InputManager.h"
#include "ActivationFunction.h"
#include <vector>

template<typename F = float>
struct NeuralNetwork {
    NeuralNetwork(const std::string& training_file, const std::string& training_labels,
                  std::vector<int> layer_sizes, std::vector<ActivationFunction<F>> functions)
    : input_manager(training_file, training_labels) {
    }
    void start_testing(const std::string& test_file, const std::string& output_file);

private:
    const size_t batch_size;
    InputManager<F> input_manager;
    std::vector<NeuronLayer<F>> layers;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
