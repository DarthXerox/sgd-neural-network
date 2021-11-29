#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "WeightLayer.h"
#include <cassert>


TEST_CASE("Transpose matrix"){
    SECTION("FIRST"){
        std::vector<std::vector<float>> weights
                {{ 1, 2, 3, 4, 5 },
                 { 6, 7, 8, 9, 10 },
                 { 11, 12, 13, 14, 15 },
                 { 16, 17, 18, 19, 20 }};

        WeightLayer<float> layer = WeightLayer<float>(4, weights.size(), weights.front().size());
        layer.set_weights(weights);

        std::vector<std::vector<float>> should_be
                {{ 1, 6, 11, 16 },
                 { 2,7,12,17 },
                 { 3,8,13,18 },
                 { 4,9,14,19 },
                 { 5,10,15,20 }};
        bool res = true;
        auto vector = layer.get_transposed_weights(weights);

        for(size_t i = 0; i < should_be.size();i++){
            for(size_t j = 0; j < should_be.front().size();j++){
                res = res && (should_be[i][j] == vector[i][j]);
            }
        }
        REQUIRE(res);
    }
    SECTION("SECOND"){
        WeightLayer<float> layer = WeightLayer<float>(4, 4,5);

        bool res = true;

        std::vector<std::vector<float>> transposed_weights = layer.get_transposed_weights();
        std::vector<std::vector<float>> weights = layer.get_weights();

        for(size_t i = 0; i < weights.size();i++){
            for(size_t j = 0; j < weights.front().size();j++){
                res = res && (weights[i][j] == transposed_weights[j][i]);
            }
        }
        REQUIRE(res);
    }


}
