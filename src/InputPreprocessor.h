#ifndef PV021_PROJECT_INPUTPREPROCESSOR_H
#define PV021_PROJECT_INPUTPREPROCESSOR_H

#include "InputManager.h"
#include <cmath>


template<typename F = float>
struct InputManager {
    InputManager(ImageInputIterator<F>&& first_iterator) {
        compute_mean();
        first_iterator.rewind();
        compute_standard_deviation();
        first_iterator.rewind();
    }


private:
    void compute_mean() {
        size_t image_count = 0;
        while (!first_iterator.is_last()) {
            float current_image_sum = 0.0f;
            for (const F& n : *first_iterator)
                current_image_sum += n;

            mean += current_image_sum / (*first_iterator).size();
            image_count++;
            ++first_iterator;
        }
        mean /= F(image_count);
    }

    void compute_standard_deviation() {
        while (!first_iterator.is_last()) {
            for (const F& n : *first_iterator) {
                standard_deviation += (n - mean) * (n - mean);
            }
            ++first_iterator;
        }
        standard_deviation /= F(image_count - 1);
        standard_deviation = std::sqrt(standard_deviation);
    }

    F mean = F(0);
    F standard_deviation;
};

#endif //PV021_PROJECT_INPUTPREPROCESSOR_H
