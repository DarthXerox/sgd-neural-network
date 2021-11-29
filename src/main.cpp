#include "WeightLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "InputManager.h"

#include <iostream>


#define SMOL_VECTORS "../data/smol_vectors.csv"
#define LOL "../data/lol.txt"

int main() {
    std::cout << "Lol, done!" << std::endl;

    auto input = InputManager<float>(SMOL_VECTORS);

    auto images = input.get_images();
    size_t images_count = images.size();
    float mean = input.get_mean();
    float d = input.get_standard_deviation();
    std::cout << "count: " << images_count << " mean: " << mean << " deviation: " << d << std::endl;


    return 0;


}
