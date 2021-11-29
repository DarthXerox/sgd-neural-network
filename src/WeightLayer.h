#ifndef PV021_PROJECT_WEIGHTLAYER_H
#define PV021_PROJECT_WEIGHTLAYER_H

#define SEED 42 // 0 and 42????

#include <vector>
#include <cstdlib>

template<typename F = float>
struct WeightLayer {
    WeightLayer(float weight_range, int lower_size, int upper_size) {
        weights = std::vector<std::vector<F>>(lower_size);

        srand(SEED);
        //srand(time(NULL));
        for (std::vector<F>& lower_neuron_weights : weights) {
            lower_neuron_weights = std::vector<F>(upper_size);
            for (F& weight : lower_neuron_weights) {
                weight = F(2) * (static_cast <F> (rand()) / static_cast <F> (RAND_MAX));
                weight -= F(1);
                weight *= weight_range;
            }
        }
        biases = std::vector<F>(upper_size); 
        for (F& bias : biases) {
            bias = F(2) * (static_cast <F> (rand()) / static_cast <F> (RAND_MAX));
            bias -= F(1);
            bias *= weight_range;
        }

        transposed_weights = get_transposed_weights(weights);
    }
    
    void correct_weights(const std::vector<std::vector<F>>& errors) {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights.front().size(); ++j) {
                weights[i][j] += errors[i][j];
            }
        }

        transposed_weights = get_transposed_weights(weights);
    }


    //WeightLayer(const std::vector<std::vector<F>>& weight_matrix, std::vector<F>&& biases);
    //void init_weights(const std::vector<std::vector<F>>& weight_matrix);

    static std::vector<std::vector<F>> get_transposed_weights(const std::vector<std::vector<F>>& weight_matrix);

    
    std::vector<F> compute_inner_potential(const std::vector<F>& input_values) {
        return WeightLayer<F>::compute_inner_potential(transposed_weights, input_values, biases);
    }
    /**
     * This could be insitu if necessary
     * @param weights must be transposed
     */
    static std::vector<F> compute_inner_potential(const std::vector<std::vector<F>>& weights,
                                                  const std::vector<F>& input_values,
                                                  const std::vector<F>& biases);

    size_t get_upper_layer_len() const {
        return transposed_weights.size();
    }

    size_t get_lower_layer_len() const {
        return weights.size();
    }

    void set_weights(const std::vector<std::vector<F>>& w) {
        this->weights = w;
    }

private:
    std::vector<std::vector<F>> weights;
    std::vector<std::vector<F>> transposed_weights;

    std::vector<F> biases; // for the upper layer

};

#endif //PV021_PROJECT_WEIGHTLAYER_H
