#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H

#include "NeuronLayer.h"
#include "InputManager.h"

template<typename F>
struct NeuralNetwork {


private:
    TrainingInputIterator training_input_iterator;
    TestInputIterator test_input_iterator
    std::vector<NeuronLayer<F>> layers;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
