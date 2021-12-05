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

            layers.push_back(WeightLayer<F>(lower_layer_size, upper_layer_size));
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

        for (int i = 0; i < batch_size; ++i) {
            backward_prop_backup.push_back(std::vector<std::vector<F>>());
            forward_prop_backup.push_back(std::vector<std::vector<F>>());
            for (int j = 0; j < layer_sizes.size() - 1; ++j) {
                backward_prop_backup[i].push_back(std::vector<F>(layer_sizes[j + 1], 0));
                forward_prop_backup[i].push_back(std::vector<F>(layer_sizes[j + 1], 0));
            }
        }

    }



    /**
     * Add momentum!!
     * maybe add adam
     */
    void train() {
        // rozdelit input na treningovy a validacny a testovaci (60 20 20 ?)
        size_t training_data_size = input_manager.get_images().size() * 0.8;
        size_t validation_data_size = input_manager.get_images().size() * 0.20;
        F error_function_sum = 0;

        //TODO 1 epocha incorrect layer size error
//        terminate called after throwing an instance of 'std::runtime_error'
//        what():  Forward prop incorrect layer size! Layer num: 0 size was: 0 and should be: 784

        while (epochs != 5) {
            std::cout << "Epoch number: " << epochs << std::endl;
            for (size_t i = 0; i < training_data_size; i += batch_size) {
                learning_rate = base_learning_rate / F(F(1) + (F(epochs) * F(training_data_size) + F(i)) / F(training_data_size));

                if(i % 10000 < batch_size){
                    std::cout << "Processed: " << i;// << std::endl;
                    std::cout << " learning rate: " << learning_rate << std::endl;
                }

//                #pragma omp parallel num_threads(8)
//                {
//                #pragma omp for

                    for (size_t j = 0; j < batch_size; j++) {
                        forward_propagation(input_manager.get_images()[i + j], j);

                    }
//                }
//                #pragma omp parallel num_threads(8)
//                {
//                #pragma omp for
                    for (int j = 0; j < batch_size; ++j) {
                        back_propagation(input_manager.get_images()[i + j], j);
                    }
//                }


                for (int k = 0; k < layers.size(); ++k) {
                    layers[k].correct_weights(all_gradients[k], bias_gradients[k], learning_rate, momentum_influence);
                }

//                #pragma omp parallel num_threads(8)
//                {
//                #pragma omp for
//                    for (int j = 0; j < layers.size(); ++j) {
//                        set_zeros(all_gradients[j]);
//                        set_zeros(bias_gradients[j]);
//                        set_zeros(backward_prop_backup[j]);
//                        set_zeros(forward_prop_backup[j]);
//                    }
//                }
                set_zeros(all_gradients);
                set_zeros(bias_gradients);
                set_zeros(backward_prop_backup);
                set_zeros(forward_prop_backup);



            }

            ++epochs;

            /**** validation was here ****/

            std::cout << "Accuracy in epochs " << epochs << " is "
                      << get_current_accuracy(training_data_size, validation_data_size) << std::endl;

            input_manager.shuffle_data(training_data_size);
        }
    }

    void start_testing(const std::string& test_file, const std::string& output_file){
        InputManager<F> test_input = InputManager<F>(test_file);
        auto test_results = std::vector<size_t>(test_input.get_training_input_count());

        //#pragma omp parallel for num_threads(NUM_THREADS)
        for (size_t i = 0; i < test_input.get_training_input_count(); ++i) {
            auto forward_prop_backups = std::vector<std::vector<F>>();
            for (unsigned long &layer_size : layer_sizes) {
                forward_prop_backups.push_back(std::vector<F>(layer_size, 0));
            }
            test_forward_propagation(test_input.get_images()[i], forward_prop_backups);
            test_results[i] = vector_max(forward_prop_backups.back());
        }
        std::ofstream test_predictions;
        test_predictions.open(output_file, std::ostream::out | std::ostream::trunc);

        for (int i = 0; i < test_input.get_training_input_count() - 1; ++i) {
            test_predictions << test_results[i] << std::endl;
        }
        test_predictions << test_results[test_input.get_training_input_count() - 1] ;
        test_predictions.close();
    }

    size_t vector_max(std::vector<F>& layer){
        F max_value = 0;
        int index = 0;
        for (int i = 0; i < layer.size(); ++i) {
            if(layer[index] < layer[i]){
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



//    F get_current_accuracy(size_t training_data_size, size_t validation_data_size) {
//        int correct = 0;
//        for (size_t i = training_data_size; i < training_data_size + validation_data_size; ++i) {
//            auto forward_prop_backups = std::vector<std::vector<F>>();
//
//            for (unsigned long &layer_size : layer_sizes) {
//                forward_prop_backups.push_back(std::vector<F>(layer_size, 0));
//            }
//
//            test_forward_propagation(input_manager.get_images()[i], forward_prop_backups);
//            size_t index = vector_max(forward_prop_backups.back());
//            //correct = index == input_manager.get_images()[i].get_label() ? correct + 1 : correct;
//            correct += size_t(index == input_manager.get_images()[i].get_label());
//
//
////            for (auto el : forward_prop_backup.back()) {
////                std::cout << el << " ";
////            }
////            std::cout << "Correct: " << correct << " now predicted: " << index << "should be: "
////                        << input_manager.get_images()[i].get_label() << std::endl;
//        }
//        return correct / F(validation_data_size);
//    }

    F get_current_accuracy(size_t training_data_size, size_t validation_data_size) {
//        int damn = 0;
        int correct = 0;
        for (size_t i = training_data_size; i < training_data_size + validation_data_size; ++i) {

            std::vector<std::vector<F>> forward;
            forward.push_back(std::vector<F>(layer_sizes[1], 0));
            forward.push_back(std::vector<F>(layer_sizes[2], 0));


            test_forward_propagation(input_manager.get_images()[i], forward);
            size_t index = vector_max(forward.back());
            //correct = index == input_manager.get_images()[i].get_label() ? correct + 1 : correct;
            correct += size_t(index == input_manager.get_images()[i].get_label());

//            if ((damn % 10000) < 10) {
//
//                for (auto el : forward.back()) {
//                    std::cout << el << " ";
//                }
//                std::cout << "Correct: " << correct << " now predicted: " << index << "should be: "
//                          << input_manager.get_images()[i].get_label() << std::endl;
//            }
//
//            damn++;
//
//            if (training_data_size + validation_data_size - 1 == i) {
//                damn++;
//            }

        }
        return F(correct) / F(validation_data_size);
    }


        private:
    /**
     * pre:
     *      forward_prop_backup.size() == layer_sizes.size()
     *      forward_prop_backup[i].size() == layer_sizes[i]
     */
    void forward_propagation(const Image<F>& input, size_t index)  {
//        if (forward_prop_backup.size() != layer_sizes.size()) {
//            throw std::runtime_error(std::string("Forward prop incorrect output vector size: "
//            + std::to_string(forward_prop_backup.size()) + " and should be: " + std::to_string(layer_sizes.size())));
//        }
//        for (size_t i = 0; i < forward_prop_backup.size(); ++i) {
//            if (forward_prop_backup[i].size() != layer_sizes[i]) {
//                throw std::runtime_error(std::string("Forward prop incorrect layer size! Layer num: "+ std::to_string(i) +
//                " size was: " + std::to_string(forward_prop_backup[i].size()) + " and should be: " + std::to_string(layer_sizes[i])));
//            }
//        }
        //#pragma omp parallel for num_threads(NUM_THREADS) // TODO this may fail

        for (size_t i = 0; i < layers.size(); ++i) {
            if(i == 0){
                layers[i].compute_inner_potential(input.get_pixels(), forward_prop_backup[index][i]);
            }
            else{
                layers[i].compute_inner_potential(forward_prop_backup[index][i - 1], forward_prop_backup[index][i]);
            }


//            if(i < layers.size() - 1){ // TODO maybe delete, bcs output is enough?
//                hidden_layer_inner_potential[i] = forward_prop_backup[i + 1];
//            }
            ActivationFunction<float>::compute(activation_functions[i], forward_prop_backup[index][i]);
        }
    }
    void test_forward_propagation(const Image<F>& input, std::vector<std::vector<F>>& forward){
        //#pragma omp parallel for num_threads(NUM_THREADS) // TODO this may fail

        for (size_t i = 0; i < layers.size(); ++i) {
            if(i == 0){
                layers[i].compute_inner_potential(input.get_pixels(), forward[i]);
            }
            else{
                layers[i].compute_inner_potential(forward[i - 1], forward[i]);
            }
            ActivationFunction<F>::compute(activation_functions[i], forward[i]);
        }

    }

    /**
     * pre:
     *      backprop_layer_output.size() == layer_sizes.size()
     *      backprop_layer_output[i].size() == layer_sizes[i]
     */
    void back_propagation(const Image<F>& input, size_t index) {
//        if (backward_prop_backup.size() != layer_sizes.size()) {
//            throw std::runtime_error(std::string("Backward prop incorrect output vector size: "
//            + std::to_string(backward_prop_backup.size())+ " and should be: " + std::to_string(layer_sizes.size())));
//        }
//        for (size_t i = 1; i < backward_prop_backup.size(); ++i) {
//            if (backward_prop_backup[i].size() != layer_sizes[i]) {
//                throw std::runtime_error(std::string("Backward prop incorrect layer size! Layer num: "+ std::to_string(i) +
//                " size was: " + std::to_string(backward_prop_backup[i].size()) + " and should be: " + std::to_string(layer_sizes[i])));
//            }
//        }
        int label = input.get_label();
        for (ssize_t l_size = layers.size() - 1; l_size >= 0; l_size--){
            if (l_size == 0)
            {//maybe not backward_prop_backup[index][l_size][j]    [j] is ?
//                #pragma omp parallel num_threads(8)
//                {
                    auto x = layers[l_size + 1].get_weights()[0].size();
//                #pragma omp for collapse(2)
                for (int i = 0; i < layers[l_size + 1].get_weights().size(); ++i) {
                    for (int j = 0; j < x; ++j) {
                        backward_prop_backup[index][l_size][j] = layers[l_size + 1].get_weight(i, j) *
                                                                 backward_prop_backup[index][l_size + 1][j] *
                                                                 ActivationFunction<F>::compute_derivative
                                                                         (activation_functions[l_size],
                                                                          forward_prop_backup[index][l_size][j]);
                    }
                }
//                }
//                #pragma omp parallel num_threads(8)
//                {
                    auto x1 = all_gradients[l_size][0].size();
//                #pragma omp for collapse(2)
                    for (int i = 0; i < all_gradients[l_size].size(); ++i) {
                        for (int j = 0; j < x1; ++j) {
                            all_gradients[l_size][i][j] += (input.get_pixels()[i]
                                                            * backward_prop_backup[index][l_size][j]) /
                                                           static_cast<F>(batch_size);
                        }
                    }
//                }
                sum_two_vectors(bias_gradients[l_size], backward_prop_backup[index][l_size]);
            }
            else if( l_size + 1 == layers.size())// the top layer is always softmax
            {
                backward_prop_backup[index][l_size] = std::vector<F>(forward_prop_backup[index][l_size]);
                backward_prop_backup[index][l_size][label] = backward_prop_backup[index][l_size][label] - 1;
//                #pragma omp parallel num_threads(8)
//                {
                    auto x = all_gradients[l_size][0].size();
//                #pragma omp for collapse(2)
                    for (int i = 0; i < all_gradients[l_size].size(); ++i) {
                        for (int j = 0; j < x; ++j) {
                            all_gradients[l_size][i][j] += (forward_prop_backup[index][l_size - 1][i]
                                                            * backward_prop_backup[index][l_size][j]) /
                                                           static_cast<F>(batch_size);
                        }
                    }
//                }
                sum_two_vectors(bias_gradients[l_size], backward_prop_backup[index][l_size]);
            }
            else  {//maybe not backward_prop_backup[index][l_size][j]    [j] is ?
//                    #pragma omp parallel num_threads(8)
//                {
                    auto x = layers[l_size + 1].get_weights()[0].size();
//                    #pragma omp for collapse(2)
                    for (int i = 0; i < layers[l_size + 1].get_weights().size(); ++i) {
                        for (int j = 0; j < x; ++j) {
                            backward_prop_backup[index][l_size][j] = layers[l_size + 1].get_weight(i, j) *
                                                                     backward_prop_backup[index][l_size + 1][j] *
                                                                     ActivationFunction<F>::compute_derivative
                                                                             (activation_functions[l_size],
                                                                              forward_prop_backup[index][l_size][j]);
                        }
                    }
//                }
//                    #pragma omp parallel num_threads(8)
//                {
                    auto x1 = all_gradients[l_size][0].size();
//                    #pragma omp for collapse(2)
                    for (int i = 0; i < all_gradients[l_size].size(); ++i) {
                        for (int j = 0; j < x1; ++j) {
                            all_gradients[l_size][i][j] += (forward_prop_backup[index][l_size - 1][i]
                                                            * backward_prop_backup[index][l_size][j]) /
                                                           static_cast<F>(batch_size);
                        }
//                    }
                }
                sum_two_vectors(bias_gradients[l_size], backward_prop_backup[index][l_size]);

            }
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

//    void correct_weights(size_t iterations){
//        for (int i = 0; i < all_gradients.size(); ++i) {
//            for (int j = 0; j < all_gradients[i].size(); ++j) {
//                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
//                    momentums[i][j][k] = -learning_rate * (all_gradients[i][j][k] + momentum_influence * momentums[i][j][k]);
////                    moments[i][j][k] = -learning_rate * all_gradients[i][j][k];
//                }
//            }
//
//
//
//            //TODO moments na biases
//            for (int j = 0; j < bias_gradients[i].size(); ++j) {
////                bias_moment[i][j] = -learning_rate * ((1- moments_influence)  * bias[i][j]
////                        + moments_influence * bias_moment[i][j]);
//                bias_moments[i][j] = -learning_rate * (bias_gradients[i][j] + momentum_influence * bias_moments[i][j]);
//            }
//            layers[i].correct_weights(momentums[i], bias_moments[i]);
//        }
//
//
////        float beta1 = 0.9,
////                        beta2 = 0.999,
////                        smol_pp = 0.00000001;
////        float alpha = 0.001;
////
////
////
////        for (int i = 0; i < all_gradients.size(); ++i) {
////            for (int j = 0; j < all_gradients[i].size(); ++j) {
////                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
////                    momentums[i][j][k] = beta1 * momentums[i][j][k] + (1.0f - beta1) * all_gradients[i][j][k];
////                    momentums[i][j][k] /= 1.0f - std::pow(beta1, iterations);
////
////                    raw_momentum[i][j][k] = beta2 * raw_momentum[i][j][k] + (1.0f - beta2) * all_gradients[i][j][k] * all_gradients[i][j][k];
////                    raw_momentum[i][j][k] /= 1.0f - std::pow(beta2, iterations);
////                }
////            }
////        }
////
////        for (int i = 0; i < all_gradients.size(); ++i) {
////            for (int j = 0; j < bias_gradients[i].size(); ++j) {
////                bias_moments[i][j] = beta1 * bias_moments[i][j] + (1.0f - beta1) * bias_gradients[i][j];
////                bias_moments[i][j] /= 1.0f - std::pow(beta1, iterations);
////
////                bias_raw_moments[i][j] = beta2 * bias_raw_moments[i][j] + (1.0f - beta2) * bias_gradients[i][j] * bias_gradients[i][j];
////                bias_raw_moments[i][j] /= 1.0f - std::pow(beta2, iterations);
////            }
////        }
////
////
////
////        // update
////        for (int i = 0; i < all_gradients.size(); ++i) {
////            for (int j = 0; j < all_gradients[i].size(); ++j) {
////                for (int k = 0; k < all_gradients[i][j].size(); ++k) {
////                    all_gradients[i][j][k] -= alpha * momentums[i][j][k] / (std::sqrt(raw_momentum[i][j][k]) + smol_pp);
////                    bias_gradients[i][j] -= alpha * bias_moments[i][j] / (std::sqrt(bias_raw_moments[i][j]) + smol_pp);
////                }
////            }
////
////            layers[i].correct_weights(all_gradients[i], bias_gradients[i]);
////        }
//
//    }

    void set_zeros(std::vector<std::vector<std::vector<F>>>& vector){

//#pragma omp parallel num_threads(8)
//        {

//#pragma omp for collapse(2)
        for (int i2 = 0; i2 < vector.size(); ++i2) {
            for (int j2 = 0; j2 < vector[i2].size(); ++j2) {
                for (int k2 = 0; k2 < vector[i2][j2].size(); ++k2) {
                    vector[i2][j2][k2] = 0;
                }
            }
        }
    }
//    }

    void set_zeros(std::vector<std::vector<F>>& vector){

//#pragma omp parallel num_threads(8)
//        {
            auto v = vector[0].size();
//#pragma omp for collapse(2)
            for (int i2 = 0; i2 < vector.size(); ++i2) {
                for (int j2 = 0; j2 < v; ++j2) {
                    vector[i2][j2] = 0;
                }
            }
        }
//    }
    void set_zeros(std::vector<F>& vector){
//#pragma omp parallel num_threads(8)
//        {
//#pragma omp for
            for (int i = 0; i < vector.size(); ++i) {
                vector[i] = 0;
            }
        }
//    }



    std::vector<F>& sum_two_vectors(std::vector<F>& fst, const std::vector<F>& snd) {
        if (fst.size() != snd.size()) {
            throw std::runtime_error("Wrong vector sizes in their sum");
        }
//        if(fst.size() < snd.size()){
//            for (size_t i = fst.size(); i < snd.size(); ++i) {
//                fst.push_back(0);
//            }
//        }
//#pragma omp parallel num_threads(8)
//        {
//#pragma omp for
            for (size_t i = 0; i < fst.size(); ++i) {
                fst[i] += snd[i];
            }
//        }
        return fst;
    }

    InputManager<F> input_manager;
    const size_t batch_size;
    float momentum_influence = 0.5; // TODO MOMENTUM_INFLUENCE 0.3
    float base_learning_rate = 0.05;
    float learning_rate = 0.05;
    //size_t input_layer_size;
    std::vector<size_t> layer_sizes;
    std::vector<FunctionType> activation_functions;
    std::vector<WeightLayer<F>> layers;

    std::vector<std::vector<F>> bias_gradients;
    std::vector<std::vector<F>> bias_moments;
    std::vector<std::vector<F>> bias_raw_moments;

    std::vector<std::vector<std::vector<F>>> momentums;
    std::vector<std::vector<std::vector<F>>> raw_momentum;

    std::vector<std::vector<std::vector<F>>> all_gradients;
    std::vector<std::vector<std::vector<F>>> backward_prop_backup;


    std::vector<std::vector<std::vector<F>>> forward_prop_backup;

    int epochs = 0;
    const int NUM_THREADS = 8;
};


#endif //PV021_PROJECT_NEURALNETWORK_H
