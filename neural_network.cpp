#include <vector>
#include <algorithm>
#include <random>

#include "neural_network.h"

#define LEARNING_RATE 0.025

/* generate a random floating point number from min to max */
double RandRange(const double &min, const double& max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// Activation functions
double Sigmoid(double input) {
    return 1.0 / (1.0 + std::exp(-input));
}

// input is sigmoid activated output 'a'
double SigmoidDerivative(double input) {
    return input * (1.0 - input);
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
    for (int i = 0; i < num_input_nodes; i++) {
        weights.push_back(RandRange(-1, 1));
    }
    ActivationFunction_ = ActivationFunction;
    ActivationFunctionDerivative_ = ActivationFunctionDerivative;
}

double Neuron::Forwards(const std::vector<double> inputs) {
    if (inputs.size() != weights.size()) {
        throw;
    }

    latest_input = inputs;

    double result = bias;
    for (int i = 0; i < inputs.size(); i++) {
        result += inputs.at(i) * weights.at(i);
    }

    latest_output = ActivationFunction_(result);

    return latest_output;
}

std::vector<double> Neuron::Backwards(const double& mean_dCost_dOutpuy) {
    double delta = mean_dCost_dOutpuy 
                   * ActivationFunctionDerivative_(latest_output);

    // Bias delta: -(learning rate * error * activation function derivative)
    // b: bias, a: output, y: target
    // ∂C/∂b = ∂z/∂b * ∂a/∂z * ∂C/∂a
    // ∂C/∂a = 2(a - y)
    // ∂a/∂z = derivative of activation function
    // ∂z/∂b = 1
    // ∂C/∂w = 2(a - y) * ∂a/∂z
    bias -= LEARNING_RATE * delta;

    // Weight change: -(learning rate * error *
    //           activation function derivative * output of previous layer)
    // w: weight, a: output, y: target
    // ∂C/∂w = ∂z/∂w * ∂a/∂z * ∂C/∂a
    // ∂C/∂a = 2(a - y)
    // ∂a/∂z = derivative of activation function
    // ∂z/∂w = a_L-1
    // ∂C/∂w = 2(a - y) * ∂a/∂z * a_L-1
    for (int i = 0; i < weights.size(); i++) {
        weights.at(i) -= LEARNING_RATE * latest_input.at(i) * delta;
    }

    // w: weight, a: output, y: target
    // ∂C/∂a_L-1 = ∂z/∂a_L-1 * ∂a/∂z * ∂C/∂a
    // ∂C/∂a = 2(a - y)
    // ∂a/∂z = derivative of activation function
    // ∂z/∂_L-1 = w
    // ∂C/∂w = 2(a - y) * ∂a/∂z * w

    std::vector<double> dCost_dInput;
    for (const double& weight : weights) {
        dCost_dInput.push_back(weight * delta);
    }
    
    // For the change in weight for previous layer:
    // ∂C/∂w_L-1 = ∂z_L-1/∂w_L-1 * ∂a_L-1/∂z * ∂C/∂a_L-1
    return dCost_dInput;
}

const void Neuron::PrintNeuron() const {
    printf("(o = %.2f; b = %.2f", latest_output, bias);
    for (int i = 0; i < weights.size(); i++) {
        printf("; w_%d = %.2f", i, weights.at(i));
    }
    printf(")");
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

void Layer::PrintLayer() const {
    for (int i = 0; i < neurons.size(); i++) {
        printf("Neuron %d: ", i);
        neurons.at(i).PrintNeuron();
        printf("\t");
    }
}

NeuralNetwork::NeuralNetwork(const int& num_inputs, const int& num_outputs, 
                const std::vector<int>& neurons_per_layer) {

    num_inputs_ = num_inputs;
    num_outputs_ = num_outputs;
        
    // First layer takes the raw inputs with no activation function
    layers.push_back(
            Layer(num_inputs, neurons_per_layer.front(), &Sigmoid, &SigmoidDerivative));

    for (int i = 1; i < neurons_per_layer.size(); i++) {
        // Each hidden layer has a number of inputs equal to the previous
        // layer's number of neurons
        layers.push_back(Layer(neurons_per_layer[i-1],
                                neurons_per_layer[i],
                                &Sigmoid, &SigmoidDerivative));
    }

    // Final layer has the last hidden layer's neuron count as the input and
    // the raw output with no activation function
    layers.push_back(
            Layer(neurons_per_layer.back(), num_outputs, &Sigmoid, &SigmoidDerivative));
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
        dCost_dOutput.push_back(2 * (last_output.at(i) - target.at(i)));
    }

    return dCost_dOutput;
}

void NeuralNetwork::PrintNetwork() const {
    printf("Neural Network Printout\n");
    printf("Number of Inputs: %d\n", num_inputs_);
    for (int i = 0; i < layers.size(); i++) {
        printf("Layer %d: ", i);
        layers.at(i).PrintLayer();
        printf("\n");
    }
    printf("Number of Outputs: %d\n", num_outputs_);
}