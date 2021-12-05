#include "WeightLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "InputManager.h"

#include <iostream>
#include <ctime>


#define SMOL_VECTORS "../data/smol_vectors.csv"
#define SMOL_LABELS "../data/smol_labels.csv"
#define SMOL_TEST_VECTORS "../data/smol_test_vectors.csv"
#define TRAIN_VECTORS "../data/fashion_mnist_train_vectors.csv"
#define TRAIN_LABELS "../data/fashion_mnist_train_labels.csv"
#define TEST_VECTORS "../data/fashion_mnist_test_vectors.csv"
#define TEST_LABELS "../data/fashion_mnist_test_labels.csv"
#define TRAIN_PREDICTIONS "../actualPredictions"
#define TRAIN_PREDICTIONS_2 "../actualPredictions2"


#define LOL "../data/lol.txt"

int main() {
    std::cout << "Lol, done!" << std::endl;



    clock_t start = clock();


    auto input = InputManager<float>(TRAIN_VECTORS);

//    auto images = input.get_images();
//    size_t images_count = images.size();
//    float mean = input.get_mean();
//    float d = input.get_standard_deviation();
//    std::cout << "count: " << images_count << " mean: " << mean << " deviation: " << d << std::endl;


    NeuralNetwork<float> neuralNetwork = NeuralNetwork<float>(TRAIN_VECTORS, TRAIN_LABELS, 16,
                                                              {64,10},
                                                              {FunctionType::Relu, FunctionType::Softmax});

//    std::vector<std::vector<float>> weights
//            {{ 1,1,1,1 },
//             { 2,2,2,2},
//             { 3,3,3,3},
//             { 4,4,4,4},
//             { 5,5,5,5},
//             { 6,6,6,6},
//             { 7,7,7,7},
//             { 8,8,8,8}};
//
//    std::vector<std::vector<float>> weights_2
//            {{ 1,1 },
//             { 2,2},
//             { 3,3},
//             { 4,4}};
//
//    neuralNetwork.set_weights({weights, weights_2});
//    neuralNetwork.start_testing(TEST_VECTORS, TRAIN_PREDICTIONS);
    neuralNetwork.train();

    neuralNetwork.start_testing(TEST_VECTORS, "../actualPredictions6");


    clock_t stop = clock();
    double elapsed = (double) (stop - start) / CLOCKS_PER_SEC;
    printf("\nTime elapsed: %.5f\n", elapsed);
    return 0;



}
