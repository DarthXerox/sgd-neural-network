#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "InputManager.h"

#include <iostream>
#include <ctime>

#define TRAIN_VECTORS "../data/fashion_mnist_train_vectors.csv"
#define TRAIN_LABELS "../data/fashion_mnist_train_labels.csv"
#define TEST_VECTORS "../data/fashion_mnist_test_vectors.csv"
#define TRAIN_PREDICTIONS "../actualPredictions"


int main() {
    clock_t start = clock();


    auto input = InputManager<float>(TRAIN_VECTORS);

    auto neuralNetwork = NeuralNetwork<float>(TRAIN_VECTORS, TRAIN_LABELS, 16,
                                                              {128,10},
                                                              {FunctionType::Relu, FunctionType::Softmax},
                                                              Optimizer::Adam);

    neuralNetwork.train();

    neuralNetwork.start_testing(TEST_VECTORS, TRAIN_PREDICTIONS);


    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);
    return 0;

}
