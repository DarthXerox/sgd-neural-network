#ifndef PV021_PROJECT_WEIGHTLAYER_H
#define PV021_PROJECT_WEIGHTLAYER_H

#include <vector>

template<typename F = float>
struct WeightLayer {
    WeightLayer(const std::vector<std::vector<F>>& weight_matrix, std::vector<F>&& b);

    void init_weights(const std::vector<std::vector<F>>& weight_matrix);

    std::vector<std::vector<F>> static get_transposed_weights(const std::vector<std::vector<F>>& weight_matrix) const;

    /**
     * This could be insitu if necessary
     * @param weights must be transposed
     */
    static std::vector<F> WeightLayer::compute_inner_potential(const std::vector<std::vector<F>>& weights,
                                                  const std::vector<F>& input_values,
                                                  const std::vector<F>& biases) const;

    size_t get_upper_layer_len() const {
        bias.size();
    }

    size_t get_lower_layer_len() const {
        weights[0].size();
    }

private:
    std::vector<std::vector<F>> weights;
    std::vector<F> bias; // for the upper layer

};

#endif //PV021_PROJECT_WEIGHTLAYER_H
