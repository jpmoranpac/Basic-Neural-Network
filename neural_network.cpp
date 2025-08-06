#include <vector>
#include <algorithm>
#include <random>

#include "neural_network.h"

#define LEARNING_RATE 0.01

/* generate a random floating point number from min to max */
double RandRange(const double &min, const double& max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

double FastSigmoid(double input) {
    // Approximate Sigmoid function:
    // 1 / (1 + |x|)
    return 1 / (1 + abs(input));
}

double Relu(double input) {
    return input < 0 ? 0 : input;
}

double Unity(double input) {
    return input;
}

Neuron::Neuron(const int& num_input_nodes,
        double (*ActivationFunction)(double),
        double (*ActivationFunctionDerivative)(double)) {
    bias = RandRange(-1, 1);
    weights.assign(num_input_nodes, RandRange(-1, 1));
    ActivationFunction_ = ActivationFunction;
    ActivationFunctionDerivative_ = ActivationFunctionDerivative;
}

double Neuron::Forwards(const std::vector<double> inputs) {
    if (inputs.size() != weights.size()) {
        throw;
    }

    double result = bias;
    for (int i = 0; i < inputs.size(); i++) {
        result += inputs.at(i) * weights.at(i);
    }

    latest_output = result;

    return ActivationFunction_(result);
}

std::vector<double> Neuron::Backwards(const double& mean_dCost_dOutpuy) {
    // Bias delta: -(learning rate * error * activation function derivative)
    // b: bias, a: output, y: target
    // ∂C/∂b = ∂z/∂b * ∂a/∂z * ∂C/∂a
    // ∂C/∂a = 2(a - y)
    // ∂a/∂z = derivative of activation function
    // ∂z/∂b = 1
    // ∂C/∂w = 2(a - y) * ∂a/∂z
    bias -= LEARNING_RATE * mean_dCost_dOutpuy 
            * ActivationFunctionDerivative_(latest_output);

    // Weight change: -(learning rate * error *
    //           activation function derivative * output of previous layer)
    // w: weight, a: output, y: target
    // ∂C/∂w = ∂z/∂w * ∂a/∂z * ∂C/∂a
    // ∂C/∂a = 2(a - y)
    // ∂a/∂z = derivative of activation function
    // ∂z/∂w = a_L-1
    // ∂C/∂w = 2(a - y) * ∂a/∂z * a_L-1
    for (double& weight : weights) {
        weight -= LEARNING_RATE * mean_dCost_dOutpuy * weight
                    * ActivationFunctionDerivative_(latest_output);
    }

    // w: weight, a: output, y: target
    // ∂C/∂a_L-1 = ∂z/∂a_L-1 * ∂a/∂z * ∂C/∂a
    // ∂C/∂a = 2(a - y)
    // ∂a/∂z = derivative of activation function
    // ∂z/∂_L-1 = w
    // ∂C/∂w = 2(a - y) * ∂a/∂z * w

    std::vector<double> dCost_dInput;
    for (const double& weight : weights) {
        dCost_dInput.push_back(mean_dCost_dOutpuy * weight
                            * ActivationFunctionDerivative_(latest_output));
    }
    
    // For the change in weight for previous layer:
    // ∂C/∂w_L-1 = ∂z_L-1/∂w_L-1 * ∂a_L-1/∂z * ∂C/∂a_L-1
    return dCost_dInput;
}


Layer::Layer(const int& num_input_nodes, const int& num_neurons,
        double (*ActivationFunction)(double),
        double (*ActivationFunctionDerivative)(double)) {
    for (int i = 0; i < num_neurons; i++) {
        neurons.push_back(Neuron(num_input_nodes,
                                    ActivationFunction,
                                    ActivationFunctionDerivative));
    }
    num_inputs = num_input_nodes;
}

std::vector<double> Layer::Forwards(const std::vector<double> inputs) {
    if (num_inputs != inputs.size()) {
        throw;
    }

    std::vector<double> output;
    for (Neuron& neuron : neurons) {
        output.push_back(neuron.Forwards(inputs));
    }

    return output;
}

// TODO: instead of returning the whole vector, process a running sum
//       of the mean error for each neuron on the previous layer
std::vector<std::vector<double>> Layer::Backwards(
            const std::vector<std::vector<double>>& dCost_dOutput) {
    // The inner vector is the dCost_dInput of a single neuron on all 
    // neurons of the previous layer
    std::vector<std::vector<double>> dCost_dInput;
    
    for (int i = 0; i < neurons.size(); i++) {
        // Calculate average dCost/dOutput for this neuron
        double mean_dCost_dOutpuy = 0.0;
        for (const auto& single_dCost : dCost_dOutput) {
            mean_dCost_dOutpuy += single_dCost.at(i);
        }
        mean_dCost_dOutpuy /= dCost_dOutput.size();

        dCost_dInput.push_back(
            neurons.at(i).Backwards(mean_dCost_dOutpuy));
    }

    return dCost_dInput;
}

NeuralNetwork::NeuralNetwork(const int& num_inputs, const int& num_outputs, 
                const std::vector<int>& neurons_per_layer) {

    // First layer takes the raw inputs with no activation function
    layers.push_back(
            Layer(num_inputs, neurons_per_layer.front(), &Unity, &Unity));

    for (int i = 1; i < neurons_per_layer.size(); i++) {
        // Each hidden layer has a number of inputs equal to the previous
        // layer's number of neurons
        layers.push_back(Layer(neurons_per_layer[i-1],
                                neurons_per_layer[i],
                                &Relu, &Relu));
    }

    // Final layer has the last hidden layer's neuron count as the input and
    // the raw output with no activation function
    layers.push_back(
            Layer(neurons_per_layer.back(), num_outputs, &Unity, &Unity));
}

std::vector<double> NeuralNetwork::Forwards(
                                        const std::vector<double>& input) {
    std::vector<double> next_input = input;
    std::vector<double> current_output;

    for (Layer& layer : layers) {
        current_output = layer.Forwards(next_input);
        next_input = current_output;
    }

    return last_output = current_output;
}

void NeuralNetwork::Backwards(const std::vector<double>& target) {
    std::vector<std::vector<double>> dCost_dOutput;
    dCost_dOutput.push_back(Calculate_dCostdOutput(target));
    std::vector<std::vector<double>> dCost_dInput;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        dCost_dInput = it->Backwards(dCost_dOutput);
        dCost_dOutput = dCost_dInput;
    }
}

std::vector<double> NeuralNetwork::CalculateError(
                                        const std::vector<double>& target) {
    if (last_output.size() != target.size()) {
        throw;
    }

    std::vector<double> error;
    for (int i = 0; i < last_output.size(); i++) {
        // Mean squared error
        error.push_back(pow(target.at(i) - last_output.at(i), 2));
    }

    return error;
}

std::vector<double> NeuralNetwork::Calculate_dCostdOutput(
                                        const std::vector<double>& target) {
    if (last_output.size() != target.size()) {
        throw;
    }

    std::vector<double> dCost_dOutput;
    for (int i = 0; i < last_output.size(); i++) {
        // Mean squared error derivative
        dCost_dOutput.push_back((target.at(i) - last_output.at(i)) *2);
    }

    return dCost_dOutput;
}