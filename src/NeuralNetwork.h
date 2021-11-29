#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H


#include "InputManager.h"
#include "ActivationFunction.h"
#include "WeightLayer.h"

#include <vector>
#include <cassert>
#include <cmath>

template<typename F = float>
struct NeuralNetwork {
    NeuralNetwork(const std::string& training_file, const std::string& training_labels, size_t batch_size,
                  std::vector<size_t> layer_sizes, std::vector<ActivationFunction<F>> functions)
    : input_manager(training_file, training_labels), batch_size(batch_size), 
      layer_sizes(layer_sizes), activation_functions(functions) {
        input_layer_size = input_manager.get_training_input_count();
        assert(layer_sizes.size() > 1);

        size_t lower_layer_size = input_layer_size;
        for (size_t i = 0; i <= layer_sizes.size(); ++i) {
            size_t upper_layer_size = layer_sizes[i];
            float weight_range = 0;

            switch (functions[i].function_type) {
                case FunctionType::Relu:
                    weight_range = std::sqrt(6 / (lower_layer_size));
                    break;
                case FunctionType::Softmax:
                    weight_range = std::sqrt(6 / (lower_layer_size + upper_layer_size));
                    break;
            }
            layers.push_back(WeightLayer<F>(weight_range, lower_layer_size, upper_layer_size));
            lower_layer_size = upper_layer_size;
        }

    }

    std::vector<F> forward_propagation(const Image<F>& input) const {
        std::vector<F> layer_inputs = input.get_pixels();
        assert(layer_inputs.size() == input_layer_size);

        for (size_t i = 0; i < layer_sizes.size(); ++i) {
            layer_inputs = layers[i].compute_inner_potential(layer_inputs);
            activation_functions[i].compute(layer_inputs);
        }

        return layer_inputs;
    }

    void train() {

        for (const Image<F>& i : input_manager.get_images()) {

        }
    }
    void start_testing(const std::string& test_file, const std::string& output_file);

private:

    std::vector<F>& sum_two_vectors(std::vector<F>& fst, const std::vector<F>& snd) {
        assert(fst.size() == snd.size());
        for (size_t i = 0; i < fst.size(); ++i) {
            fst[i] += snd[i];
        }
        return fst;
    }

    InputManager<F> input_manager;
    const size_t batch_size;
    size_t input_layer_size;
    std::vector<size_t> layer_sizes;
    std::vector<ActivationFunction<F>> activation_functions;
    std::vector<WeightLayer<F>> layers;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
