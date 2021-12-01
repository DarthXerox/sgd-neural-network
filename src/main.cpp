#include "WeightLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "InputManager.h"

#include <iostream>


#define SMOL_VECTORS "../data/smol_vectors.csv"
#define SMOL_LABELS "../data/smol_labels.csv"
#define TRAIN_VECTORS "../data/fashion_mnist_test_vectors.csv"
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


    NeuralNetwork<float> neuralNetwork = NeuralNetwork<float>(SMOL_VECTORS, SMOL_LABELS, 16,
                                                              {64,10},
                                                              {FunctionType::Relu, FunctionType::Softmax});



    WeightLayer<float> layer = WeightLayer<float>(4, 5,6);




    return 0;


}
