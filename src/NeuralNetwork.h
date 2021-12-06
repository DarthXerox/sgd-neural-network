#ifndef PV021_PROJECT_NEURALNETWORK_H
#define PV021_PROJECT_NEURALNETWORK_H


#include "InputManager.h"
#include "ActivationFunction.h"
#include "WeightLayer.h"

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>


enum struct Optimizer {
    Adam,
    RMSProp,
    MomentumOnly
};

template<typename F = float>
struct NeuralNetwork {
    NeuralNetwork(const std::string& training_file, const std::string& training_labels, size_t batch_size,
                  std::vector<size_t>&& non_input_layer_sizes,
                  std::vector<FunctionType>&& functions, Optimizer o)
    : input_manager(training_file, training_labels), batch_size(batch_size), activation_functions(std::move(functions)),
    optimizer(o) {
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
    }


    void train() {
        size_t training_data_size = input_manager.get_images().size() * 0.9;
        size_t validation_data_size = input_manager.get_images().size() * 0.1;

        int max_epochs = optimizer == Optimizer::Adam ? 6 : 8;


        while (epochs != max_epochs) {
            std::cout << "Epoch number: " << epochs << std::endl;
            for (size_t i = 0; i < training_data_size; i += batch_size) {
                if (optimizer == Optimizer::MomentumOnly)
                    learning_rate = base_learning_rate / F(F(1) + (F(epochs) * F(training_data_size) + F(i)) / F(training_data_size));


                auto backward_prop_backup = std::vector<std::vector<F>>(),
                        forward_prop_backup = std::vector<std::vector<F>>();
                for (unsigned long &layer_size : layer_sizes) {
                    backward_prop_backup.push_back(std::vector<F>(layer_size));
                    forward_prop_backup.push_back(std::vector<F>(layer_size));
                }

                for (size_t j = 0; j < batch_size; j++) {
                    forward_propagation(input_manager.get_images()[i + j], forward_prop_backup);
                    back_propagation(input_manager.get_images()[i + j].get_label(), forward_prop_backup,
                                     backward_prop_backup);
                    compute_gradients(forward_prop_backup, backward_prop_backup);
                }

                switch (optimizer) {
                    case Optimizer::Adam:
                        perform_papo_weight_correction(i + batch_size + epochs * training_data_size);
                        break;
                    case Optimizer::RMSProp:
                        perform_rmsprop_weight_correction();
                        break;
                    case Optimizer::MomentumOnly:
                        perform_momentum_weight_correction();
                        break;
                }

                // clearing gradients
                for (int j = 0; j < all_gradients.size(); ++j) {
                    for (int k = 0; k < all_gradients.at(j).size(); ++k) {
                        for (int l = 0; l < all_gradients.at(j).at(k).size(); ++l) {
                            all_gradients.at(j).at(k).at(l) = 0;
                        }
                    }
                }
                for (int j = 0; j < bias_gradients.size(); ++j) {
                    for (int k = 0; k < bias_gradients.at(j).size(); ++k) {
                        bias_gradients.at(j).at(k) = 0;
                    }
                }
            }

            ++epochs;
            std::cout << "Accuracy in epochs " << epochs << " is "
                      << get_current_accuracy(training_data_size, validation_data_size) << std::endl;

            input_manager.shuffle_data(training_data_size);

            // clearing all momentum
            momentums.clear();
            raw_momentum.clear();
            bias_raw_moments.clear();
            //all_gradients.clear();
            //bias_gradients.clear();
            bias_moments.clear();
            for (int j = 0; j < layer_sizes.size() - 1; ++j) {
                momentums.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
                raw_momentum.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
                //all_gradients.push_back(std::vector<std::vector<F>>(layer_sizes[j], std::vector<F>(layer_sizes[j+1], 0)));
                //bias_gradients.push_back(std::vector<F>(layer_sizes[j+1], 0));
                bias_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
                bias_raw_moments.push_back(std::vector<F>(layer_sizes[j+1], 0));
            }
        }
    }

    void start_testing(const std::string& test_file, const std::string& output_file){
        InputManager<F> test_input = InputManager<F>(test_file);
        auto test_results = std::vector<size_t>(test_input.get_training_input_count());

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
    
    void compute_gradients(const std::vector<std::vector<F>>& forward_prop,
                                 const std::vector<std::vector<F>>& backward_prop) {
        for (int layer_index = 0; layer_index < layers.size(); ++layer_index) {
           if(layer_index + 1 == layers.size()){
               for (size_t j = 0; j < layers[layer_index].get_lower_layer_len(); ++j) {
                   for (size_t k = 0; k < layers[layer_index].get_upper_layer_len(); ++k) {
                       all_gradients[layer_index][j][k] += (backward_prop[layer_index + 1][k] * forward_prop[layer_index][j]) / F(batch_size);
                   }
               }
               for (int j = 0; j < bias_gradients[layer_index].size(); ++j) {
                   bias_gradients[layer_index][j] += backward_prop[layer_index + 1][j] / F(batch_size);
               }
           }
           else{
               for (size_t j = 0; j < layers[layer_index].get_lower_layer_len(); ++j) {
                   for (size_t k = 0; k < layers[layer_index].get_upper_layer_len(); ++k) {
                       all_gradients[layer_index][j][k] += (backward_prop[layer_index + 1][k]
                               * ActivationFunction<F>::compute_derivative(activation_functions[layer_index], forward_prop[layer_index + 1][k])
                               * forward_prop[layer_index][j]) / F(batch_size);
                   }
               }
               for (int j = 0; j < bias_gradients[layer_index + 1].size(); ++j) {
                   bias_gradients[layer_index + 1][j] += (backward_prop[layer_index + 1][j]
                           * ActivationFunction<F>::compute_derivative(activation_functions[layer_index], forward_prop[layer_index + 1][j])) / F(batch_size);
               }
           }
        }
        
    }


    F get_current_accuracy(size_t training_data_size, size_t validation_data_size) {
        int correct = 0;
        for (size_t i = training_data_size; i < training_data_size + validation_data_size; ++i) {
            auto forward_prop_backup = std::vector<std::vector<F>>();

            for (unsigned long &layer_size : layer_sizes) {
                forward_prop_backup.push_back(std::vector<F>(layer_size));
            }

            forward_propagation(input_manager.get_images()[i], forward_prop_backup);
            size_t index = vector_max(forward_prop_backup.back());
            correct += size_t(index == input_manager.get_images()[i].get_label());

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
        }

    }

// TODO should we keep old momentum?????
    void perform_rmsprop_weight_correction() {
        F p = 0.9,
                n = 0.001,
                smol_pp = 0.00001;

        // gradients
        for (int i = 0; i < all_gradients.size(); ++i) {
            for (int j = 0; j < all_gradients.at(i).size(); ++j) {
                for (int k = 0; k < all_gradients.at(i).at(j).size(); ++k) {
                    raw_momentum.at(i).at(j).at(k) = p * raw_momentum.at(i).at(j).at(k)
                            + (1 - p) * all_gradients.at(i).at(j).at(k) * all_gradients.at(i).at(j).at(k);
                }
            }
        }

        for (int i = 0; i < all_gradients.size(); ++i) {
            for (int j = 0; j < all_gradients.at(i).size(); ++j) {
                for (int k = 0; k < all_gradients.at(i).at(j).size(); ++k) {
                    all_gradients.at(i).at(j).at(k) = (-n / std::sqrt(raw_momentum.at(i).at(j).at(k) + smol_pp))
                             * all_gradients.at(i).at(j).at(k);
                }
            }
        }

        // bias gradients
        for (int i = 0; i < bias_gradients.size(); ++i) {
           for (int j = 0; j < bias_gradients.at(i).size(); ++j) {
               bias_raw_moments.at(i).at(j) = p * bias_raw_moments.at(i).at(j)
                       + (1 - p) * bias_gradients.at(i).at(j) * bias_gradients.at(i).at(j);
           }
        }

        for (int i = 0; i < bias_gradients.size(); ++i) {
           for (int j = 0; j < bias_gradients.at(i).size(); ++j) {
               bias_gradients.at(i).at(j) = (-n / std::sqrt(bias_raw_moments.at(i).at(j) + smol_pp))
                       * bias_gradients.at(i).at(j);
           }

           // update
           layers.at(i).correct_weights(all_gradients.at(i), bias_gradients.at(i));
        }
    }

    void perform_papo_weight_correction(size_t iterations) {
        F beta1 = 0.9,
            beta2 = 0.999,
            smol_pp = 0.000001,
            beta1_pow = std::pow(beta1, F(iterations)),
            beta2_pow = std::pow(beta2, F(iterations));
        F alpha = 0.001;

        for (int i = 0; i < all_gradients.size(); ++i) {
            for (int j = 0; j < all_gradients.at(i).size(); ++j) {
                for (int k = 0; k < all_gradients.at(i).at(j).size(); ++k) {
                    momentums.at(i).at(j).at(k) = beta1 * momentums.at(i).at(j).at(k) + (1.0f - beta1) * all_gradients.at(i).at(j).at(k);
                    raw_momentum.at(i).at(j).at(k) = beta2 * raw_momentum.at(i).at(j).at(k) + (1.0f - beta2) * all_gradients.at(i).at(j).at(k) * all_gradients.at(i).at(j).at(k);
                }
            }
        }

        for (int i = 0; i < bias_gradients.size(); ++i) {
            for (int j = 0; j < bias_gradients.at(i).size(); ++j) {
                bias_moments.at(i).at(j) = beta1 * bias_moments.at(i).at(j) + (1.0f - beta1) * bias_gradients.at(i).at(j);
                bias_raw_moments.at(i).at(j) = beta2 * bias_raw_moments.at(i).at(j) + (1.0f - beta2) * bias_gradients.at(i).at(j) * bias_gradients.at(i).at(j);
            }
        }

        // update
        for (int i = 0; i < all_gradients.size(); ++i) {
            for (int j = 0; j < all_gradients.at(i).size(); ++j) {
                for (int k = 0; k < all_gradients.at(i).at(j).size(); ++k) {
                    all_gradients.at(i).at(j).at(k) = -alpha * (momentums.at(i).at(j).at(k) / (1.0f - beta1_pow + smol_pp))
                            / (std::sqrt(raw_momentum.at(i).at(j).at(k) / (1.0f - beta2_pow + smol_pp)) + smol_pp);
                }
            }

        }

        for (int i = 0; i < bias_gradients.size(); ++i) {
           for (int j = 0; j < bias_gradients.at(i).size(); ++j) {
               bias_gradients.at(i).at(j) = -alpha * (bias_moments.at(i).at(j) / (1.0f - beta1_pow + smol_pp))
                       / (std::sqrt(bias_raw_moments.at(i).at(j)/ (1.0f - beta2_pow + smol_pp)) + smol_pp);
           }

            layers.at(i).correct_weights(all_gradients.at(i), bias_gradients.at(i));
        }
    }

    void perform_momentum_weight_correction(){
        for (int i = 0; i < all_gradients.size(); ++i) {
            for (int j = 0; j < all_gradients[i].size(); ++j) {
                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
                    momentums[i][j][k] = -learning_rate * all_gradients[i][j][k] + momentum_influence * momentums[i][j][k];
                }
            }



            //TODO moments na biases
            for (int j = 0; j < bias_gradients[i].size(); ++j) {
                bias_moments[i][j] = -learning_rate * bias_gradients[i][j] + momentum_influence * bias_moments[i][j];
            }
            layers[i].correct_weights(momentums[i], bias_moments[i]);
        }
    }



    std::vector<F>& sum_two_vectors(std::vector<F>& fst, const std::vector<F>& snd) {
        if (fst.size() != snd.size()) {
            throw std::runtime_error("Wrong vector sizes in their sum");
        }
        for (size_t i = 0; i < fst.size(); ++i) {
            fst[i] += snd[i];
        }
        return fst;
    }

    InputManager<F> input_manager;
    const size_t batch_size;
    float momentum_influence = 0.55; // TODO MOMENTUM_INFLUENCE 0.3
                    float base_learning_rate = 0.03;
                    float learning_rate = 0.03;
    std::vector<size_t> layer_sizes;
    std::vector<FunctionType> activation_functions;
    std::vector<WeightLayer<F>> layers;
    std::vector<std::vector<F>> bias_gradients;
    std::vector<std::vector<F>> bias_moments;
    std::vector<std::vector<F>> bias_raw_moments;
    std::vector<std::vector<std::vector<F>>> momentums;
    std::vector<std::vector<std::vector<F>>> raw_momentum;
    std::vector<std::vector<std::vector<F>>> all_gradients;
    int epochs = 0;
    Optimizer optimizer;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
