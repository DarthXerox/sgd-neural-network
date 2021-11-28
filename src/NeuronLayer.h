#ifndef PV021_PROJECT_NEURONLAYER_H
#define PV021_PROJECT_NEURONLAYER_H

#include "WeightLayer.h"
#include "ActivationFunction.h"

template<typename F = float>
struct NeuronLayer {
    void correct_weights(const std::vector<std::vector<F>>& new_weights);
    std::vector<F> perform_activation(std::vector<F> inputs);
    std::vector<std::vector<F>> perform_back_propagation(std::vector<F> from_upper, ActivationFunction<F>);

private:
    WeightLayer<F> weight_layer;
    ActivationFunction<F> activation_function;
};


#endif //PV021_PROJECT_NEURONLAYER_H
