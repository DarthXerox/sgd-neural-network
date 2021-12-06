#ifndef PV021_PROJECT_WEIGHTLAYER_H
#define PV021_PROJECT_WEIGHTLAYER_H


#include <vector>
#include <cstdlib>
#include <random>
#include <exception>
#include <stdexcept>

std::mt19937 mt(42);


template<typename F = float>
struct WeightLayer {
   WeightLayer(F weight_range, int lower_size, int upper_size) {
       weights = std::vector<std::vector<F>>(lower_size);
       biases = std::vector<F>(upper_size);

       if(upper_size == 10){
           std::normal_distribution<F> dist{0, 1};
           for (std::vector<F>& lower_neuron_weights : weights) {
               lower_neuron_weights = std::vector<F>(upper_size);
               for (F& weight : lower_neuron_weights) {
                   weight = dist(mt) * std::sqrt(2.0f / static_cast<F>(lower_size));
               }
           }
           for (F& bias : biases) {
               bias = dist(mt) * std::sqrt(2.0f / static_cast<F>(lower_size));
           }

       }
       else{
           F limit = std::sqrt(6.0f / (static_cast<F>(lower_size + upper_size)));
           std::uniform_real_distribution<F> dist(-limit, limit);
           for (std::vector<F>& lower_neuron_weights : weights) {
               lower_neuron_weights = std::vector<F>(upper_size);
               for (F& weight : lower_neuron_weights) {
                   weight = dist(mt);
               }
           }
           for (F& bias : biases) {
               bias = dist(mt);
           }

       }

       transposed_weights = std::vector<std::vector<F>>(upper_size, std::vector<F>(lower_size, 0));

       transpose_weights();
   }

   void correct_weights(const std::vector<std::vector<F>>& errors, const std::vector<F>& bias_error) {
       for (size_t i = 0; i < weights.size(); ++i) {
           for (size_t j = 0; j < weights.front().size(); ++j) {
               weights[i][j] += errors[i][j];
           }
       }
       for (int i = 0; i < bias_error.size(); ++i) {
           biases[i] += bias_error[i];
       }

       transpose_weights();
   }



   void transpose_weights() {
       for (size_t i = 0; i < weights.size(); ++i) {
           for (size_t j = 0; j < weights.front().size(); ++j) {
               transposed_weights.at(j).at(i) = weights.at(i).at(j);
           }
       }
   }



   /**
    * This could be insitu if necessary
    * @param output can contain garbage, is meant to be overwritten
    * @param weights must be transposed
    */
   void compute_inner_potential(const std::vector<F>& input_values, std::vector<F>& output){
       WeightLayer<F>::vector_matrix_mul(input_values, transposed_weights, output);
       for (size_t i = 0; i < output.size(); ++i) {
           output[i] += biases[i];
       }
   }

   /**
    * preconditions:
    *      input_values.size() == matrix[0].size()
    *      output.size() == matrix.size()
    */
   static void vector_matrix_mul(const std::vector<F>& input_values, const std::vector<std::vector<F>>& matrix,
                          std::vector<F>& output) {

           for (size_t i = 0; i < matrix.size(); ++i) {
               output[i] = F(0);
               for (size_t j = 0; j < input_values.size(); ++j) {
                   output[i] += matrix[i][j] * input_values[j];
               }
           }
   }

   size_t get_upper_layer_len() const {
       return transposed_weights.size();
   }

   size_t get_lower_layer_len() const {
       return weights.size();
   }

   void set_weights(const std::vector<std::vector<F>>& w) {
       weights = w;
       get_transposed_weights();
   }


   std::vector<std::vector<F>> get_transposed_weights()const {
       return transposed_weights;
   }

   F& get_weight(size_t i, size_t j){
       return weights[i][j];
   }


private:
   std::vector<std::vector<F>> weights;
   std::vector<std::vector<F>> transposed_weights;

   std::vector<F> biases;

};

#endif //PV021_PROJECT_WEIGHTLAYER_H
