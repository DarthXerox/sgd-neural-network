#ifndef PV021_PROJECT_ACTIVATIONFUNCTIONCALCULATOR_H
#define PV021_PROJECT_ACTIVATIONFUNCTIONCALCULATOR_H

#include <vector>
#include <cmath>
#include <cassert>
enum struct FunctionType {
    Relu,
    Softmax
};

template<typename F = float>
struct ActivationFunction {
    static std::vector<F>& compute(FunctionType type, std::vector<F>& x) {
        switch (type) {
            case FunctionType::Relu:
                for (F& val : x) {
                    val = val < F(0) ? F(0) : val;
                }
                return x;
                break;
            case FunctionType::Softmax: {
                F maximum = x[0];
                for (size_t i = 1; i < x.size(); i++) {
                   if (x[i] > maximum) {
                       maximum = x[i];
                   }
                }
                F sum = 0;
                for (size_t i = 0; i < x.size(); i++) {
                   sum += std::exp(x[i] - maximum);
                }

                for (size_t i = 0; i < x.size(); i++) {
                   x[i] = std::exp(x[i] - maximum) / (std::max(sum, 10e-6f));
                }
                return x;
            }
        }
        return x;
    }
    static F compute_derivative(FunctionType type, F x) {
        switch (type) {
            case FunctionType::Relu: {
                return x <= F(0) ? F(0) : F(1);
            }
            case FunctionType::Softmax: {
                throw std::runtime_error("We don't derive softmax kekw");
            }

        }
        return x;
    }
};


#endif //PV021_PROJECT_ACTIVATIONFUNCTIONCALCULATOR_H
