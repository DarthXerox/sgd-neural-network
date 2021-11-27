#include "WeightLayer.h"
#include <vector>


WeightLayer::WeightLayer(const std::vector<std::vector<F>>& weight_matrix,
                         std::vector<F>&& b) : bias(std::move(b)) {
    init_weights(weight_matrix);
}


void WeightLayer::init_weights(const std::vector<std::vector<F>>& weight_matrix)  {
    for (const std::vector<F>& row : weight_matrix) {
        std::vector<F> new_row(row.size());
        for (size_t i = 0; i < row.size(); ++i) {
            new_row[i] = row[i];
        }
        weights.push_back(new_row);
    }
}


std::vector<std::vector<F>> static WeightLayer::get_transposed_weights(
        const std::vector<std::vector<F>>& weight_matrix) const {
    if (weight_matrix.empty()) {
        return std::vector<std::vector<F>>();
    }

    size_t col_len = weight_matrix[0].size();
    std::vector<std::vector<F>> result(col_len);

    #pragma omp parallel for
    for (size_t j = 0; j < weight_matrix.size(); ++j) {
        std::vector<T> new_row(weight_matrix.size());
        for (size_t i = 0; i < col_len; ++i) {
            new_row[i] = weight_matrix[j][i];
        }
        result[j] = new_row;
    }

    return result;
}


template<typename F = float>
static std::vector<F> WeightLayer::compute_inner_potential(const std::vector<std::vector<F>>& weights,
                                                           const std::vector<F>& input_values,
                                                           const std::vector<F>& biases) const {
    if (weights.empty() || input_values.size() != weights[0].size() || biases.size() != weights.size()) {
        return std::vector<F>(); // throw??
    }

    std::vector<F> result(weights.size());
    for (size_t i = 0; i < input_values; ++i) {
        for (size_t j = 0; j < weights.size(); ++j) {
            result[i] += weights[i][j] * input_values[j];
        }
        result[i] += bias[i];
    }

    return result;
}
