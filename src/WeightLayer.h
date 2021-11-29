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

        transposed_weights = WeightLayer<F>::get_transposed_weights(weights);
    }


    //WeightLayer(const std::vector<std::vector<F>>& weight_matrix, std::vector<F>&& biases);
    //void init_weights(const std::vector<std::vector<F>>& weight_matrix);

    std::vector<std::vector<F>> get_transposed_weights(std::vector<std::vector<F>>& weight_matrix){
        if (weight_matrix.empty()) {
            return std::vector<std::vector<F>>();
        }

        size_t col_len = weight_matrix[0].size();
        std::vector<std::vector<F>> result(col_len);

//        #pragma omp parallel for
        for (size_t j = 0; j < col_len; ++j) {
            std::vector<F> new_row(weight_matrix.size());
            for (size_t i = 0; i < weight_matrix.size(); ++i) {
                new_row[i] = weight_matrix[i][j];
            }
            result[j] = new_row;
        }

        return result;
    }


    /**
     * This could be insitu if necessary
     * @param weights must be transposed
     */
    std::vector<F> compute_inner_potential(const std::vector<F>& input_values){
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

    size_t get_upper_layer_len() const {
        return transposed_weights.size();
    }

    size_t get_lower_layer_len() const {
        return weights.size();
    }

    void set_weights(const std::vector<std::vector<F>>& w) {
        weights = w;
    }

    std::vector<std::vector<F>> get_weights(){
        return weights;
    }

    std::vector<std::vector<F>> get_transposed_weights(){
        return transposed_weights;
    }

private:
    std::vector<std::vector<F>> weights;
    std::vector<std::vector<F>> transposed_weights;

    std::vector<F> biases; // for the upper layer

};

#endif //PV021_PROJECT_WEIGHTLAYER_H
