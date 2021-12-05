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
            raw_momentum.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
            all_gradients.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
            bias_gradients.push_back(std::vector<F>(layer_sizes[j+1], 0));
            bias_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
            bias_raw_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
        }
        //learning_rate = 0.05;
        //base_learning_rate = learning_rate;
//        prepare_backup();
    }



    /**
     * Add momentum!!
     * maybe add adam
     */
    void train() {
        // rozdelit input na treningovy a validacny a testovaci (60 20 20 ?)
        size_t training_data_size = input_manager.get_images().size() * 0.8;
        size_t validation_data_size = input_manager.get_images().size() * 0.2;
        F error_function_sum = 0;

        //TODO 1 epocha incorrect layer size error
//        terminate called after throwing an instance of 'std::runtime_error'
//        what():  Forward prop incorrect layer size! Layer num: 0 size was: 0 and should be: 784

        while (epochs != 8) {
            std::cout << "Epoch number: " << epochs << std::endl;
            for (size_t i = 0; i < training_data_size; i += batch_size) {
                learning_rate = base_learning_rate / F(F(1) + (F(epochs) * F(training_data_size) + F(i)) / F(training_data_size));

                if(i % 10000 < 1){
                    std::cout << "Processed: " << i;// << std::endl;
                    std::cout << " learning rate: " << learning_rate << std::endl;
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
                //backward_prop_backup.front().clear();

                if (i == 30272) {
                   // std::cout << "here we go?" << std::endl;
                }
                for (size_t j = 0; j < batch_size; j++) {
                    if (j == 1) {
                     //   std::cout << "here we go?" << std::endl;
                    }
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

                // TODO kuknut v slajdoch
                // take average of the batch
//                for (auto &vec : backward_batch_average) {
//                    for (auto &el : vec) {
//                        el /= F(batch_size);
//                    }
//                }
//                for (auto &vec : forward_batch_average) {
//                    for (auto &el : vec) {
//                        el /= F(batch_size);
//                    }
//                }

                // change learning rate

                determine_gradients(forward_batch_average, backward_batch_average);
                correct_weights(i + batch_size);
            }

            ++epochs;

            /**** validation was here ****/

            std::cout << "Accuracy in epochs " << epochs << " is "
                      << get_current_accuracy(training_data_size, validation_data_size) << std::endl;

            input_manager.shuffle_data(training_data_size);

            momentums.clear();
            all_gradients.clear();
            bias_gradients.clear();
            bias_moments.clear();
            for (int j = 0; j < layer_sizes.size() - 1; ++j) {
                momentums.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
                raw_momentum.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
                all_gradients.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
                bias_gradients.push_back(std::vector<F>(layer_sizes[j+1], 0));
                bias_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
                bias_raw_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
            }
        }
    }

    void start_testing(const std::string& test_file, const std::string& output_file){
        InputManager<F> test_input = InputManager<F>(test_file);
        auto test_results = std::vector<size_t>(test_input.get_training_input_count());

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
        test_predictions.open(output_file, std::ostream::out | std::ostream::trunc);

        for (int i = 0; i < test_input.get_training_input_count() - 1; ++i) {
            test_predictions << test_results[i] << std::endl;
        }
        test_predictions << test_results[test_input.get_training_input_count() - 1] ;
        test_predictions.close();
    }

    // som upravil
    size_t vector_max(std::vector<F>& layer){
        F max_value = 0;
        int index = 0;
        for (int i = 0; i < layer.size(); ++i) {
            if(max_value < layer[i]){
                max_value = layer[i];
                index = i;
            }
        }
        return index;
    }


    void set_weights(std::vector<std::vector<std::vector<F>>> weights){
        for (size_t i = 0; i < layers.size(); i++){
            layers[i].set_weights(weights[i]);
        }
    }

    void determine_gradients(const std::vector<std::vector<F>>& forward_batch_average,
                             const std::vector<std::vector<F>>& backward_batch_average) {
        for (int layer_index = 0; layer_index < layers.size(); ++layer_index) {
           if(layer_index + 1 == layers.size()){
               for (size_t j = 0; j < layers[layer_index].get_lower_layer_len(); ++j) {
                   for (size_t k = 0; k < layers[layer_index].get_upper_layer_len(); ++k) {
                       //vaha z i do j = back z j * output z i
                       assert(backward_batch_average[layer_index + 1].size() == 10);
                       all_gradients[layer_index][j][k] = backward_batch_average[layer_index + 1][k]
                               * forward_batch_average[layer_index][j];
                       if (std::isnan(all_gradients[layer_index][j][k])) {
                          throw std::runtime_error("aaaach");
                        }
                   }
               }
               /*for (int j = 0; j < bias_gradients[layer_index].size(); ++j) {
                   bias_gradients[layer_index][j] = backward_batch_average[layer_index + 1][j];
               }*/
           }
           else{
               for (size_t j = 0; j < layers[layer_index].get_lower_layer_len(); ++j) {
                   for (size_t k = 0; k < layers[layer_index].get_upper_layer_len(); ++k) {
                       //vaha z i do j = back z j * output z i
                       all_gradients[layer_index][j][k] = backward_batch_average[layer_index + 1][k]
                               * ActivationFunction<F>::compute_derivative
                               (activation_functions[layer_index], forward_batch_average[layer_index + 1][k])
                               * forward_batch_average[layer_index][j];
                   }
               }
               for (int j = 0; j < bias_gradients[layer_index + 1].size(); ++j) {
                   bias_gradients[layer_index + 1][j] = backward_batch_average[layer_index + 1][j]
                           * ActivationFunction<F>::compute_derivative
                           (activation_functions[layer_index], forward_batch_average[layer_index + 1][j]);
               }
           }
        }


//        for (int layer_index = 1; layer_index < layers.size(); ++layer_index) {
//            if (layer_index == layers.size()){
//                for (size_t j = 0; j < layers[layer_index - 1].get_lower_layer_len(); ++j) {
//                    for (size_t k = 0; k < layers[layer_index-1].get_upper_layer_len(); ++k) {
//                        // weight from i to j = back from j * output from i
//                        all_gradients[layer_index - 1][j][k] = backward_batch_average[layer_index][k]
//                                * forward_batch_average[layer_index - 1][j];
//                    }
//                }
//                for (int j = 0; j < bias_gradients[layer_index].size(); ++j) {
//                    bias_gradients[layer_index][j] = backward_batch_average[layer_index][j];
//                }
//            }
//            else {
//                for (size_t j = 0; j < layers[layer_index - 1].get_lower_layer_len(); ++j) {
//                    for (size_t k = 0; k < layers[layer_index-1].get_upper_layer_len(); ++k) {
//                        // weight from i to j = back from j * output from i
//                        all_gradients[layer_index - 1][j][k] = backward_batch_average[layer_index][k]
//                                * ActivationFunction<F>::compute_derivative(activation_functions[layer_index - 1],
//                                                                            forward_batch_average[layer_index][k])
//                                * forward_batch_average[layer_index - 1][j];
//                    }
//                }
//                for (int j = 0; j < bias_gradients[layer_index].size(); ++j) {
//                    bias_gradients[layer_index][j] = backward_batch_average[layer_index][j]
//                            * ActivationFunction<F>::compute_derivative(activation_functions[layer_index - 1],
//                                                                        forward_batch_average[layer_index][j]);
//                }
//            }
//        }


    }

    F get_current_accuracy(size_t training_data_size, size_t validation_data_size) {
        int damn = 0;
        int correct = 0;
        for (size_t i = training_data_size; i < training_data_size + validation_data_size; ++i) {
            auto forward_prop_backup = std::vector<std::vector<F>>();

            for (unsigned long &layer_size : layer_sizes) {
                forward_prop_backup.push_back(std::vector<F>(layer_size));
            }

            forward_propagation(input_manager.get_images()[i], forward_prop_backup);
            size_t index = vector_max(forward_prop_backup.back());
            //correct = index == input_manager.get_images()[i].get_label() ? correct + 1 : correct;
            correct += size_t(index == input_manager.get_images()[i].get_label());

//            if ((damn % 100) < 10) {
//
//            for (auto el : forward_prop_backup.back()) {
//                std::cout << el << " ";
//            }
//            std::cout << "Correct: " << correct << " now predicted: " << index << "should be: "
//                        << input_manager.get_images()[i].get_label() << std::endl;
//            }

            damn++;

            if (training_data_size + validation_data_size - 1 == i) {
                damn++;
            }

        }
        return correct / F(validation_data_size);
    }

private:
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
            if (std::isnan(forward_prop_backup[i + 1][0])) {
              throw std::runtime_error("aaaach");
            }
//            if(i < layers.size() - 1){ // TODO maybe delete, bcs output is enough?
//                hidden_layer_inner_potential[i] = forward_prop_backup[i + 1];
//            }
            ActivationFunction<F>::compute(activation_functions[i], forward_prop_backup[i + 1]);
            if (std::isnan(forward_prop_backup[i + 1][0])) {
              throw std::runtime_error("aaaach");
            }

        }
    }

    /**
     * pre:
     *      backprop_layer_output.size() == layer_sizes.size()
     *      backprop_layer_output[i].size() == layer_sizes[i]
     */
    void back_propagation(size_t label, const  std::vector<std::vector<F>>& forward_prop_backup,
                          std::vector<std::vector<F>>& backprop_layer_output) {
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
                if (std::isnan(backprop_layer_output[l_size][label])) {
                  throw std::runtime_error("aaaach");
                }
                for (size_t i = 0; i < forward_prop_backup[l_size - 1].size(); i++){
                    F out_sum = 0;
                    for (size_t j = 0; j < backprop_layer_output[l_size].size(); ++j) {
                        out_sum += backprop_layer_output[l_size][j] * layers[l_size - 1].get_weight(i,j);
                    }
                    backprop_layer_output[l_size - 1][i] = out_sum;
                    if (std::isnan(backprop_layer_output[l_size - 1][i])) {
                      throw std::runtime_error("aaaach");
                    }
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

    void correct_weights(size_t iterations){
        for (int i = 0; i < all_gradients.size(); ++i) {
            for (int j = 0; j < all_gradients[i].size(); ++j) {
                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
//                    moments[i][j][k] = -learning_rate * ((1- moments_influence) * all_gradients[i][j][k]
//                              + moments_influence * moments[i][j][k]);
                    momentums[i][j][k] = -learning_rate * all_gradients[i][j][k] + momentum_influence * momentums[i][j][k];
//                    moments[i][j][k] = -learning_rate * all_gradients[i][j][k];
                }
            }



            //TODO moments na biases
            for (int j = 0; j < bias_gradients[i].size(); ++j) {
//                bias_moment[i][j] = -learning_rate * ((1- moments_influence)  * bias[i][j]
//                        + moments_influence * bias_moment[i][j]);
                bias_moments[i][j] = -learning_rate * bias_gradients[i][j] + momentum_influence * bias_moments[i][j];
            }
            layers[i].correct_weights(momentums[i], bias_moments[i]);
        }


//        float beta1 = 0.9,
//                        beta2 = 0.999,
//                        smol_pp = 0.00000001;
//        float alpha = 0.001;
//
//
//
//        for (int i = 0; i < all_gradients.size(); ++i) {
//            for (int j = 0; j < all_gradients[i].size(); ++j) {
//                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
//                    momentums[i][j][k] = beta1 * momentums[i][j][k] + (1.0f - beta1) * all_gradients[i][j][k];
//                    momentums[i][j][k] /= 1.0f - std::pow(beta1, iterations);
//
//                    raw_momentum[i][j][k] = beta2 * raw_momentum[i][j][k] + (1.0f - beta2) * all_gradients[i][j][k] * all_gradients[i][j][k];
//                    raw_momentum[i][j][k] /= 1.0f - std::pow(beta2, iterations);
//                }
//            }
//        }
//
//        for (int i = 0; i < all_gradients.size(); ++i) {
//            for (int j = 0; j < bias_gradients[i].size(); ++j) {
//                bias_moments[i][j] = beta1 * bias_moments[i][j] + (1.0f - beta1) * bias_gradients[i][j];
//                bias_moments[i][j] /= 1.0f - std::pow(beta1, iterations);
//
//                bias_raw_moments[i][j] = beta2 * bias_raw_moments[i][j] + (1.0f - beta2) * bias_gradients[i][j] * bias_gradients[i][j];
//                bias_raw_moments[i][j] /= 1.0f - std::pow(beta2, iterations);
//            }
//        }
//
//
//
//        // update
//        for (int i = 0; i < all_gradients.size(); ++i) {
//            for (int j = 0; j < all_gradients[i].size(); ++j) {
//                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
//                    all_gradients[i][j][k] -= alpha * momentums[i][j][k] / (std::sqrt(raw_momentum[i][j][k]) + smol_pp);
//                    bias_gradients[i][j] -= alpha * bias_moments[i][j] / (std::sqrt(bias_raw_moments[i][j]) + smol_pp);
//                }
//            }
//
//            layers[i].correct_weights(all_gradients[i], bias_gradients[i]);
//        }

    }



    std::vector<F>& sum_two_vectors(std::vector<F>& fst, const std::vector<F>& snd) {
        if (fst.size() != snd.size()) {
            throw std::runtime_error("Wrong vector sizes in their sum");
        }
//        if(fst.size() < snd.size()){
//            for (size_t i = fst.size(); i < snd.size(); ++i) {
//                fst.push_back(0);
//            }
//        }
        for (size_t i = 0; i < fst.size(); ++i) {
            fst[i] += snd[i];
        }
        return fst;
    }

    InputManager<F> input_manager;
    const size_t batch_size;
//    float momentum_influence = 0.7; // TODO  10 epoch -> 84.5%
//    float base_learning_rate = 0.05;
//    float learning_rate = 0.05;
//    float momentum_influence = 0.5; // TODO 8 epoch -> 86.9% (po 10 87.3% ale 128neuronov)
//                float base_learning_rate = 0.05;
//                float learning_rate = 0.05;
    float momentum_influence = 0.55; // TODO MOMENTUM_INFLUENCE 0.3
                    float base_learning_rate = 0.03;
                    float learning_rate = 0.03;
//    float momentum_influence = 0.57; // TODO MOMENTUM_INFLUENCE 0.3
//                        float base_learning_rate = 0.03;
//                        float learning_rate = 0.03;
    //size_t input_layer_size;
    std::vector<size_t> layer_sizes;
    std::vector<FunctionType> activation_functions;
    std::vector<WeightLayer<F>> layers;
    //std::vector<std::vector<F>> hidden_layer_inner_potential;
    //std::vector<std::vector<F>> forward_prop_backup;
    //std::vector<std::vector<F>> output;
    std::vector<std::vector<F>> bias_gradients;
    std::vector<std::vector<F>> bias_moments;
    std::vector<std::vector<F>> bias_raw_moments;

    std::vector<std::vector<std::vector<F>>> momentums;
    std::vector<std::vector<std::vector<F>>> raw_momentum;

    std::vector<std::vector<std::vector<F>>> all_gradients;
    std::vector<F> forward_prop_batch;

    int epochs = 0;
    const int NUM_THREADS = 8;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
