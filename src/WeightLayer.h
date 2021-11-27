#ifndef PV021_PROJECT_WEIGHTLAYER_H
#define PV021_PROJECT_WEIGHTLAYER_H

#include <vector>

template<typename F = float>
struct WeightLayer {
    WeightLayer(const std::vector<std::vector<F>>& weight_matrix, std::vector<F>&& b);
    void init_weights(const std::vector<std::vector<F>>& weight_matrix);

    static std::vector<std::vector<F>> get_transposed_weights(const std::vector<std::vector<F>>& weight_matrix);

    /**
     * This could be insitu if necessary
     * @param weights must be transposed
     */
    static std::vector<F> compute_inner_potential(const std::vector<std::vector<F>>& weights,
                                                  const std::vector<F>& input_values,
                                                  const std::vector<F>& biases);

    size_t get_upper_layer_len() const {
        return biases.size();
    }

    size_t get_lower_layer_len() const {
        return weights[0].size();
    }

private:
    std::vector<std::vector<F>> weights;
    std::vector<F> biases; // for the upper layer

};

#endif //PV021_PROJECT_WEIGHTLAYER_H
