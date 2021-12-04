#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H


#include "InputManager.h"
#include "ActivationFunction.h"
#include "WeightLayer.h"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

template<typename F = float>
struct NeuralNetwork {
    NeuralNetwork(const std::string& training_file, const std::string& training_labels, size_t batch_size,
                  std::vector<size_t>&& non_input_layer_sizes,
                  std::vector<FunctionType>&& functions)
    : input_manager(training_file, training_labels), batch_size(batch_size), activation_functions(std::move(functions)) {
        layer_sizes.push_back(input_manager.get_pixel_per_image_count());
        layer_sizes.insert(this->layer_sizes.end(), std::make_move_iterator(non_input_layer_sizes.begin()),
                                 std::make_move_iterator(non_input_layer_sizes.end()));

        size_t lower_layer_size = layer_sizes[0];
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            size_t upper_layer_size = layer_sizes[i];
            float weight_range = 0;

            switch (activation_functions[i - 1]) {
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
        for (int j = 0; j < layer_sizes.size() - 1; ++j) {
            momentums.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
            all_gradients.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
            bias_gradients.push_back(std::vector<F>(layer_sizes[j+1], 0));
            bias_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
        }
        learning_rate = 0.01;
//        prepare_backup();
    }



    /**
     * Add momentum!!
     * maybe add adam
     */
    void train() {
        // rozdelit input na treningovy a validacny a testovaci (60 20 20 ?)
        size_t training_data_size = input_manager.get_images().size()*0.99;
//        size_t validation_data_size = input_manager.get_images().size() * 0.20;
        F error_function_sum = 0;

        //TODO 1 epocha incorrect layer size error
//        terminate called after throwing an instance of 'std::runtime_error'
//        what():  Forward prop incorrect layer size! Layer num: 0 size was: 0 and should be: 784
        while (epochs != 6) {
            std::cout << epochs << std::endl;
            // OR while (error_function.result > nejaka_mala_hodnota) + pocitat epochy
            for (size_t i = 0; i < training_data_size; ++i) {
                if(i % 10000 < 16){
                    std::cout << i << std::endl;
                }
                // prepare 16 matrices
                auto backward_prop_backup = std::vector<std::vector<F>>(),
                        backward_batch_average = std::vector<std::vector<F>>(),
                        forward_batch_average = std::vector<std::vector<F>>(),
                        forward_prop_backup = std::vector<std::vector<F>>();


                for (unsigned long &layer_size : layer_sizes) {
                    backward_prop_backup.push_back(std::vector<F>(layer_size));
                    forward_prop_backup.push_back(std::vector<F>(layer_size));
                    backward_batch_average.push_back(std::vector<F>(layer_size));
                    forward_batch_average.push_back(std::vector<F>(layer_size));
                }
                backward_prop_backup.front().clear();

                for (size_t j = 0; j < batch_size; j++) {
                    forward_propagation(input_manager.get_images()[i + j], forward_prop_backup);
                    back_propagation(input_manager.get_images()[i + j].get_label(), forward_prop_backup,
                                     backward_prop_backup);

                    // add to the batch sum
                    for (size_t k = 0; k < backward_prop_backup.size(); ++k) {
                        sum_two_vectors(backward_batch_average[k], backward_prop_backup[k]);
                    }

                    for (size_t k = 0; k < forward_prop_backup.size(); ++k) {
                        sum_two_vectors(forward_batch_average[k], forward_prop_backup[k]);
                    }
                }
                i += batch_size;

                // take average of the batch
                for (auto &vec : backward_batch_average) {
                    for (auto &el : vec) {
                        el /= F(batch_size);
                    }
                }
                for (auto &vec : forward_batch_average) {
                    for (auto &el : vec) {
                        el /= F(batch_size);
                    }
                }

                // change learning rate
                // CORRECT weights + biases including learning rate



                for (int layer_index = 1; layer_index < layers.size(); ++layer_index) {

                    if(layer_index == layers.size()){
                        for (size_t j = 0; j < layers[layer_index - 1].get_lower_layer_len(); ++j) {
                            for (size_t k = 0; k < layers[layer_index-1].get_upper_layer_len(); ++k) {
                                //vaha z i do j = back z j * output z i
                                all_gradients[layer_index - 1][j][k] = backward_batch_average[layer_index][k]
                                        * forward_batch_average[layer_index - 1][j];
                            }
                        }
                        for (int j = 0; j < bias_gradients[layer_index].size(); ++j) {
                            bias_gradients[layer_index][j] = backward_batch_average[layer_index][j];
                        }
                    }
                    else{
                        for (size_t j = 0; j < layers[layer_index - 1].get_lower_layer_len(); ++j) {
                            for (size_t k = 0; k < layers[layer_index-1].get_upper_layer_len(); ++k) {
                                //vaha z i do j = back z j * output z i
                                all_gradients[layer_index - 1][j][k] = backward_batch_average[layer_index][k]
                                        * ActivationFunction<F>::compute_derivative
                                        (activation_functions[layer_index - 1], forward_batch_average[layer_index][k])
                                        * forward_batch_average[layer_index - 1][j];
                            }
                        }
                        for (int j = 0; j < bias_gradients[layer_index].size(); ++j) {
                            bias_gradients[layer_index][j] = backward_batch_average[layer_index][j]
                                    * ActivationFunction<F>::compute_derivative
                                    (activation_functions[layer_index - 1], forward_batch_average[layer_index][j]);
                        }
                    }
                }
                correct_weights(all_gradients, momentums, bias_gradients, bias_moments);
            }



            ++epochs;

//            if (epochs > 3) {
//                F old_error = error_function_sum;
//                error_function_sum = 0;
//
//                for (size_t i = training_data_size; i < training_data_size + validation_data_size; ++i) {
//                    auto forward_prop_backup = std::vector<std::vector<F>>();
//
//
//                    for (unsigned long &layer_size : layer_sizes) {
//                        forward_prop_backup.push_back(std::vector<F>(layer_size));
//                    }
//
//                    forward_propagation(input_manager.get_images()[i], forward_prop_backup);
//
//                    error_function_sum += std::log(
//                            forward_prop_backup.back()[input_manager.get_images()[i].get_label()]);
//                }
//                error_function_sum /= -1 * validation_data_size;
//                if (epochs > 4) {
//                    if (old_error < error_function_sum) {
//                        return;
//                    }
//                }
//            }
            input_manager.shuffle_data(training_data_size - 1);
        }
    }

    void start_testing(const std::string& test_file, const std::string& output_file){
        InputManager<F> test_input = InputManager<F>(test_file);
        auto test_results = std::vector<std::string>(test_input.get_training_input_count());

        //#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t i = 0; i < test_input.get_training_input_count(); ++i) {
            auto forward_prop_backup = std::vector<std::vector<F>>();
            for (unsigned long &layer_size : layer_sizes) {
                forward_prop_backup.push_back(std::vector<F>(layer_size));
            }
            forward_propagation(test_input.get_images()[i], forward_prop_backup);
            test_results[i] = vector_max(forward_prop_backup.back());
        }
        std::ofstream test_predictions;
        test_predictions.open (output_file);

        for (int i = 0; i < test_input.get_training_input_count() - 1; ++i) {
            test_predictions << test_results[i] + "\n";
        }
        test_predictions << test_results[test_input.get_training_input_count() - 1] ;
        test_predictions.close();
    }

    std::string vector_max(std::vector<F>& layer){
        F max_value;
        int index = 0;
        for (int i = 0; i < layer.size(); ++i) {
            if(max_value < layer[i]){
                max_value = layer[i];
                index = i;
            }
        }
        return std::to_string(index);
    }


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
     *      forward_prop_backup.size() == layer_sizes.size()
     *      forward_prop_backup[i].size() == layer_sizes[i]
     */
    void forward_propagation(const Image<F>& input, std::vector<std::vector<F>>& forward_prop_backup)  {
        if (forward_prop_backup.size() != layer_sizes.size()) {
            throw std::runtime_error(std::string("Forward prop incorrect output vector size: "
            + std::to_string(forward_prop_backup.size()) + " and should be: " + std::to_string(layer_sizes.size())));
        }
        for (size_t i = 0; i < forward_prop_backup.size(); ++i) {
            if (forward_prop_backup[i].size() != layer_sizes[i]) {
                throw std::runtime_error(std::string("Forward prop incorrect layer size! Layer num: "+ std::to_string(i) +
                " size was: " + std::to_string(forward_prop_backup[i].size()) + " and should be: " + std::to_string(layer_sizes[i])));
            }
        }
        //#pragma omp parallel for num_threads(NUM_THREADS) // TODO this may fail
        forward_prop_backup[0] = input.get_pixels();
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].compute_inner_potential(forward_prop_backup[i], forward_prop_backup[i + 1]);
//            if(i < layers.size() - 1){ // TODO maybe delete, bcs output is enough?
//                hidden_layer_inner_potential[i] = forward_prop_backup[i + 1];
//            }
            ActivationFunction<float>::compute(activation_functions[i], forward_prop_backup[i + 1]);
        }
    }

    /**
     * pre:
     *      backprop_layer_output.size() == layer_sizes.size()
     *      backprop_layer_output[i].size() == layer_sizes[i]
     */
    void back_propagation(size_t label, const  std::vector<std::vector<F>>& forward_prop_backup,
                          std::vector<std::vector<F>>& backprop_layer_output) {
        //backprop-example-proof.png prekreslit v painte a pochopit vystup back_propagation
        if (backprop_layer_output.size() != layer_sizes.size()) {
            throw std::runtime_error(std::string("Backward prop incorrect output vector size: "
            + std::to_string(backprop_layer_output.size())+ " and should be: " + std::to_string(layer_sizes.size())));
        }
        for (size_t i = 1; i < backprop_layer_output.size(); ++i) {
            if (backprop_layer_output[i].size() != layer_sizes[i]) {
                throw std::runtime_error(std::string("Backward prop incorrect layer size! Layer num: "+ std::to_string(i) +
                " size was: " + std::to_string(backprop_layer_output[i].size()) + " and should be: " + std::to_string(layer_sizes[i])));
            }
        }

        for (size_t l_size = layers.size(); l_size > 1; l_size--){
            if (l_size == layers.size()){ // the top layer is always softmax
                backprop_layer_output[l_size] = std::vector<F>(forward_prop_backup[l_size]);
                backprop_layer_output[l_size][label] -= 1;
                for (size_t i = 0; i < forward_prop_backup[l_size - 1].size(); i++){
                    F out_sum = 0;
                    for (size_t j = 0; j < backprop_layer_output[l_size].size(); ++j) {
                        out_sum += backprop_layer_output[l_size][j] * layers[l_size - 1].get_weight(i,j);
                    }
                    backprop_layer_output[l_size - 1][i] = out_sum;
                }
            }
//            else  {
//                for(size_t i = 0; i < forward_prop_backup[l_size - 1].size(); i++){
//                    F out_sum = 0;
//                    for(size_t j = 0; j < forward_prop_backup[l_size].size();j++){
//                        out_sum += backprop_layer_output[l_size][j] *
//                                   ActivationFunction<float>::compute_derivative
//                                (activation_functions[l_size - 1], hidden_layer_inner_potential[l_size - 1][j]) *
//                                   layers[l_size - 1].get_weight(i,j);
//                    }
//                    backprop_layer_output[l_size - 1][i] = out_sum;
//                }
//            }
        }

    }


//    void prepare_backup() {
//        //hidden_layer_inner_potential.push_back(std::vector<F>());
//        forward_prop_backup.push_back(std::vector<F>());
//
//       /* for (size_t i = 0; i < layer_sizes.size(); ++i) {
//            output.push_back(std::vector<F>(layer_sizes[i]));
//        }*/
//
//        hidden_layer_inner_potential = std::vector<F>(layer_sizes.size() - 1);
//
//        for (unsigned long & layer_size : layer_sizes) {
//            forward_prop_backup.push_back(std::vector<F>(layer_size));
//        }
//    }
//
//    std::vector<F>& get_last_layer(){
//        return forward_prop_backup[forward_prop_backup.size() - 1];
//    }
//

    void correct_weights(std::vector<std::vector<std::vector<F>>>& gradients,
                         std::vector<std::vector<std::vector<F>>>& momentum,
                         std::vector<std::vector<F>>& bias,
                         std::vector<std::vector<F>>& bias_moment){
        for (int i = 0; i < gradients.size(); ++i) {
            for (int j = 0; j < gradients[i].size(); ++j) {
                for (int k = 0; k < gradients[i][j].size(); ++k) {
                    momentum[i][j][k] = -learning_rate * gradients[i][j][k] + momentum_influence * momentum[i][j][k];
//                    momentum[i][j][k] = -learning_rate * gradients[i][j][k];
                }
            }
            //TODO momentum na biases
            for (int j = 0; j < bias[i].size(); ++j) {
                bias_moment[i][j] = -learning_rate * bias[i][j] + momentum_influence * bias_moment[i][j];
            }
            layers[i].correct_weights(momentum[i], bias_moment[i]);
        }

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
    float momentum_influence = 0.9;
    //size_t input_layer_size;
    std::vector<size_t> layer_sizes;
    std::vector<FunctionType> activation_functions;
    std::vector<WeightLayer<F>> layers;
    //std::vector<std::vector<F>> hidden_layer_inner_potential;
    //std::vector<std::vector<F>> forward_prop_backup;
    //std::vector<std::vector<F>> output;
    std::vector<std::vector<F>> bias_gradients;
    std::vector<std::vector<F>> bias_moments;
    std::vector<std::vector<std::vector<F>>> momentums;
    std::vector<std::vector<std::vector<F>>> all_gradients;
    std::vector<F> forward_prop_batch;
    float learning_rate;
    int epochs = 0;
    const int NUM_THREADS = 8;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
