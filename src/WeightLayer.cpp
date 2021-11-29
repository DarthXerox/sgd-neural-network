#include "WeightLayer.h"
#include <vector>

/*
template<typename F>
WeightLayer<F>::WeightLayer(const std::vector<std::vector<F>>& weight_matrix,
                         std::vector<F>&& b) : biases(std::move(b)) {
    init_weights(weight_matrix);
}


template<typename F>
void WeightLayer<F>::init_weights(const std::vector<std::vector<F>>& weight_matrix)  {
    for (const std::vector<F>& row : weight_matrix) {
        std::vector<F> new_row(row.size());
        for (size_t i = 0; i < row.size(); ++i) {
            new_row[i] = row[i];
        }
        weights.push_back(new_row);
    }
}
*/

template<typename F>
std::vector<std::vector<F>> WeightLayer<F>::get_transposed_weights(
        const std::vector<std::vector<F>>& weight_matrix) {
    if (weight_matrix.empty()) {
        return std::vector<std::vector<F>>();
    }

    size_t col_len = weight_matrix[0].size();
    std::vector<std::vector<F>> result(col_len);

    #pragma omp parallel for
    for (size_t j = 0; j < weight_matrix.size(); ++j) {
        std::vector<F> new_row(weight_matrix.size());
        for (size_t i = 0; i < col_len; ++i) {
            new_row[i] = weight_matrix[j][i];
        }
        result[j] = new_row;
    }

    return result;
}


template<typename F>
std::vector<F> WeightLayer<F>::compute_inner_potential(const std::vector<std::vector<F>>& weights,
                                                           const std::vector<F>& input_values,
                                                           const std::vector<F>& biases) {
    if (weights.empty() || input_values.size() != weights[0].size() || biases.size() != weights.size()) {
        return std::vector<F>(); // throw??
    }

    std::vector<F> result(weights.size());
    for (size_t i = 0; i < input_values; ++i) {
        for (size_t j = 0; j < weights.size(); ++j) {
            result[i] += weights[i][j] * input_values[j];
        }
        result[i] += biases[i];
    }

    return result;
}
