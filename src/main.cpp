#include "WeightLayer.h"
#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "InputManager.h"

#include <iostream>


#define SMOL_VECTORS "../data/smol_vectors.csv"
#define SMOL_LABELS "../data/smol_labels.csv"

#define LOL "../data/lol.txt"

int main() {
    std::cout << "Lol, done!" << std::endl;

    /*auto input = InputManager<float>(SMOL_VECTORS, SMOL_LABELS);

    auto images = input.get_images();
    size_t images_count = images.size();
    float mean = input.get_mean();
    float d = input.get_standard_deviation();
    std::cout << "count: " << images_count << " mean: " << mean << " deviation: " << d << std::endl;
*/

    WeightLayer<float> layer = WeightLayer<float>(4, 5,6);




    return 0;


}
