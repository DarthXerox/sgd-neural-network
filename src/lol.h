#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H


#include "InputManager.h"
#include "ActivationFunction.h"
#include "WeightLayer.h"

#include <vector>
#include <cassert>
#include <cmath>

template<typename F = float>
struct NeuralNetwork {
    NeuralNetwork(const std::string& training_file, const std::string& training_labels, size_t batch_size,
                  std::vector<size_t> layer_sizes,
                  std::vector<FunctionType> functions)
    : input_manager(training_file, training_labels), batch_size(batch_size),
      layer_sizes(layer_sizes), activation_functions(functions) {
        input_layer_size = input_manager.get_pixel_per_image_count();
        assert(layer_sizes.size() > 1);

        size_t lower_layer_size = input_layer_size;
        for (size_t i = 0; i < layer_sizes.size(); ++i) {
            size_t upper_layer_size = layer_sizes[i];
            float weight_range = 0;

            switch (functions[i]) {
                case FunctionType::Relu:
                    weight_range = std::sqrt(6 / (float(lower_layer_size)));
                    break;
                case FunctionType::Softmax:
                    weight_range = std::sqrt(6 / (float(lower_layer_size) + float(upper_layer_size)));
                    break;
            }
            layers.push_back(WeightLayer<F>(weight_range, lower_layer_size, upper_layer_size));
            lower_layer_size = upper_layer_size;
        }
        learning_rate = 0.001;
        prepare_backup();
    }



    void train() {

        for (const Image<F>& i : input_manager.get_images()) {

            //


            for(size_t j = 0; j < batch_size; j++){
                forward_propagation(i);
                back_propagation(i.get_label());

            }


            break;
        }
    }
    void start_testing(const std::string& test_file, const std::string& output_file);

    void set_weights(std::vector<std::vector<std::vector<F>>> weights){
        for (size_t i = 0; i < layers.size(); i++){
            layers[i].set_weights(weights[i]);
        }
    }

private:
    std::vector<F> forward_propagation_single_layer(const std::vector<F>& input, size_t layer_index) const {
        std::vector<F> mutable_input = input;
        mutable_input = layers[layer_index].compute_inner_potential(mutable_input);
        ActivationFunction<float>::compute(activation_functions[layer_index], mutable_input);

        return mutable_input;
    }


    /**
     * pre:
     *      forward_prop_backup.size() == layer_sizes.size() + 1
     */
    void forward_propagation(const Image<F>& input)  {
        //#pragma omp parallel for num_threads(NUM_THREADS) // TODO this may fail
        forward_prop_backup[0] = input.get_pixels();
            for (size_t i = 1; i <= layer_sizes.size(); ++i) {
                layers[i-1].compute_inner_potential(forward_prop_backup[i-1], forward_prop_backup[i]);
                if(i != layer_sizes.size()){
                    hidden_layer_inner_potential[i - 1] = forward_prop_backup[i];
                }
                ActivationFunction<float>::compute(activation_functions[i-1], forward_prop_backup[i]);
            }
    }

    void back_propagation(F label){
        //backprop-example-proof.png prekreslit v painte a pochopit vystup back_propagation

        for(size_t l_size = layers.size(); l_size > 0; l_size--){
            if(l_size == layers.size()){ // the top layer is always softmax
                for(size_t i = 0; i < forward_prop_backup[l_size - 1].size(); i++){
                    F out_sum = 0;
                    for(size_t j = 0; j < forward_prop_backup[l_size].size();j++){

                        out_sum += label == j ? (forward_prop_backup[l_size][j] - 1)
                                * layers[l_size - 1].get_weight(i,j):
                                   forward_prop_backup[l_size][j] * layers[l_size - 1].get_weight(i,j);
                    }
                    output[l_size - 1][i] = out_sum;
                }
            }
            if(l_size < layers.size()){
                for(size_t i = 0; i < forward_prop_backup[l_size - 1].size(); i++){
                    F out_sum = 0;
                    for(size_t j = 0; j < forward_prop_backup[l_size].size();j++){
                        out_sum += output[l_size][j] *
                                   ActivationFunction<float>::compute_derivative
                                (activation_functions[l_size - 1], hidden_layer_inner_potential[l_size - 1][j]) *
                                   layers[l_size - 1].get_weight(i,j);
                    }
                    output[l_size - 1][i] = out_sum;
                }
            }
        }

    }


    void prepare_backup() {
        //hidden_layer_inner_potential.push_back(std::vector<F>());
        forward_prop_backup.push_back(std::vector<F>());

        for (size_t i = 0; i < layer_sizes.size(); ++i) {
            output.push_back(std::vector<F>(layer_sizes[i]));
        }

        hidden_layer_inner_potential = std::vector<F>(layer_sizes.size() - 1);

        for (unsigned long & layer_size : layer_sizes) {
            forward_prop_backup.push_back(std::vector<F>(layer_size));
        }
    }

    std::vector<F>& get_last_layer(){
        return forward_prop_backup[forward_prop_backup.size() - 1];
    }


    std::vector<F>& sum_two_vectors(std::vector<F>& fst, const std::vector<F>& snd) {
        if(fst.size() < snd.size()){
            for (size_t i = fst.size(); i < snd.size(); ++i) {
                fst.push_back(0);
            }
        }
        for (size_t i = 0; i < fst.size(); ++i) {
            fst[i] += snd[i];
        }
        return fst;
    }

    InputManager<F> input_manager;
    const size_t batch_size;
    size_t input_layer_size;
    std::vector<size_t> layer_sizes;
    std::vector<FunctionType> activation_functions;
    std::vector<WeightLayer<F>> layers;
    std::vector<std::vector<F>> hidden_layer_inner_potential;
    std::vector<std::vector<F>> forward_prop_backup;
    std::vector<std::vector<F>> output;
    std::vector<F> forward_prop_batch;
    float learning_rate;
    int epochs = 0;
    const int NUM_THREADS = 8;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
