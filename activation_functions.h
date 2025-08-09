#include <algorithm>

// Activation functions
double Sigmoid(double input) {
    return 1.0 / (1.0 + std::exp(-input));
}

double SigmoidDerivative(double input) {
    return input * (1.0 - input);
}

double Relu(double input) {
    return input < 0 ? 0 : input;
}

double Unity(double input) {
    return input;
}