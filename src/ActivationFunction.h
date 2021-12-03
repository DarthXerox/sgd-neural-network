#ifndef PV021_PROJECT_ACTIVATIONFUNCTIONCALCULATOR_H
#define PV021_PROJECT_ACTIVATIONFUNCTIONCALCULATOR_H

#include <vector>
#include <cmath>

enum struct FunctionType {
    Relu,
    Softmax
};

template<typename F = float>
struct ActivationFunction {
    //explicit ActivationFunction(FunctionType f):function_type(f){};
    //virtual ~ActivationFunction() = default;
    static std::vector<F>& compute(FunctionType type, std::vector<F>& x) {
        switch (type) {
            case FunctionType::Relu:
                for (F& val : x) {
                    val = val < F(0) ? F(0) : val;
                }
                return x;
                break;
            case FunctionType::Softmax: {
                F exp_sum = F(0);
                for (size_t i = 0; i < x.size(); ++i) {
                    exp_sum += std::exp(x[i]);
                }

                for (size_t i = 0; i < x.size(); ++i) {
                    x[i] = std::exp(x[i]) / exp_sum;
                }

                return x;
                //break;
            }
        }
        return x;
    }
    static F compute_derivative(FunctionType type, F& x) {
        switch (type) {
            case FunctionType::Relu: {
                return x <= F(0) ? F(0) : F(1);
            }
            case FunctionType::Softmax: {
                return x;
            }

        }
        return x;
    }
    //virtual static TOut compute_derivative(TIn x) = 0;

    //FunctionType function_type;
};

//
//template<typename F = float>
//struct ReluActivationFunction : public ActivationFunction<F> {
//
//    ReluActivationFunction():ActivationFunction<F>(FunctionType::Relu) {
//    }
//
//    std::vector<F>& compute(std::vector<F>& x) override {
//        for (F& val : x) {
//            val = val < F(0) ? F(0) : val;
//        }
//        return x;
//    }
//
//    /*
//    static std::vector<F>& compute_derivative(std::vector<F>& x) override {
//        for (F& val : x) {
//            val = val < (0) ? F(0) : F(1);
//        }
//        return x;
//    }*/
//};
//
//
//template<typename F = float>
//struct SoftmaxActivationFunction : public ActivationFunction<F> {
//
//    SoftmaxActivationFunction():ActivationFunction<F>(FunctionType::Softmax) {}
//
//    std::vector<F>& compute(std::vector<F>& x) override {
//        F exp_sum = F(0);
//        for (size_t i = 0; i < x.size(); ++i) {
//            exp_sum += std::exp(x[i]);
//        }
//
//        for (size_t i = 0; i < x.size(); ++i) {
//            x[i] = std::exp(x[i]) / exp_sum;
//        }
//
//        return x;
//    }
///*
//    static std::vector<F>& compute_derivative(std::vector<F>& x) override {
//        compute(x);
//        for (F& val : x) {
//            val = val * (F(1) - val);
//        }
//    }*/
//};

template<typename F = float>
struct CrossEntropyErrorFunction {
    static F compute_derivative(F x) {
        return F(1) / x;
    }

};

#endif //PV021_PROJECT_ACTIVATIONFUNCTIONCALCULATOR_H
