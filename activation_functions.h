#include <algorithm>

/// @brief Sigmoid function, y = 1 / (1 + e^-x)
/// @param input x
/// @return y
double Sigmoid(double input) {
    return 1.0 / (1.0 + std::exp(-input));
}

/// @brief Derivative of the Sigmoid function, y' = sig(x) * (1 - sig(x)), where
///        sig(x) is the output of the Sigmoid function
/// @param sigmoid_x sig(x), NOT x
/// @return y'
double SigmoidDerivative(double sigmoid_x) {
    return sigmoid_x * (1.0 - sigmoid_x);
}

/// @brief Rectified linear unit, y = x for x > 0 and y = 0 for x < 0
/// @param input x
/// @return y
double Relu(double input) {
    return input < 0 ? 0 : input;
}

/// @brief No activation function, y = x
/// @param input x
/// @return y
double Unity(double input) {
    return input;
}