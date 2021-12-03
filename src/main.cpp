#include "WeightLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "InputManager.h"

#include <iostream>


#define SMOL_VECTORS "../data/smol_vectors.csv"
#define SMOL_LABELS "../data/smol_labels.csv"
#define TRAIN_VECTORS "../data/fashion_mnist_train_vectors.csv"
#define TRAIN_LABELS "../data/fashion_mnist_train_labels.csv"
#define TEST_VECTORS "../data/fashion_mnist_test_vectors.csv"
#define TEST_LABELS "../data/fashion_mnist_test_labels.csv"
#define TRAIN_PREDICTIONS "../actualPredictions"


#define LOL "../data/lol.txt"

int main() {
    std::cout << "Lol, done!" << std::endl;

    auto input = InputManager<float>(TRAIN_VECTORS);

//    auto images = input.get_images();
//    size_t images_count = images.size();
//    float mean = input.get_mean();
//    float d = input.get_standard_deviation();
//    std::cout << "count: " << images_count << " mean: " << mean << " deviation: " << d << std::endl;


    NeuralNetwork<float> neuralNetwork = NeuralNetwork<float>(SMOL_VECTORS, SMOL_LABELS, 2,
                                                              {64,10},
                                                              {FunctionType::Relu, FunctionType::Softmax});



    neuralNetwork.train();


    return 0;


}
