#define CATCH_CONFIG_MAIN

#include "../catch.hpp"

#include "WeightLayer.h"
#include <cassert>


TEST_CASE("Transpose matrix"){
    std::vector<std::vector<float>> weights
                                         {{ 1, 2, 3, 4, 5 },
                                         { 6, 7, 8, 9, 10 },
                                         { 11, 12, 13, 14, 15 },
                                         { 16, 17, 18, 19, 20 }};
    WeightLayer<float> layer = new WeightLayer<float>(weights);

    std::vector<std::vector<float>> should_be
                                        {{ 1, 6, 11, 16 },
                                        { 2,7,12,17 },
                                        { 3,8,13,18 },
                                        { 4,9,14,19 },
                                        { 5,10,15,20 }};

    REQUIRE(layer.get_transposed_weights(weights) == should_be);

}

}