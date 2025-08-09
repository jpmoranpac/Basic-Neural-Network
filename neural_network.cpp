#include <vector>
#include <random>
#include <stdexcept>

#include "neural_network.h"
#include "activation_functions.h"

#define LEARNING_RATE 0.025

// Generate a random number between min and max
double RandRange(const double &min, const double& max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// =======================================
// Constructors
// =======================================

NeuralNetwork::NeuralNetwork(const int& num_inputs, const int& num_outputs, 
                             const std::vector<int>& neurons_per_layer):
                             num_inputs_(num_inputs), num_outputs_(num_outputs)
                             {
        
    // First layer
    layers.push_back(Layer(num_inputs, neurons_per_layer.front(),
                           &Sigmoid, &SigmoidDerivative));

    for (int i = 1; i < neurons_per_layer.size(); i++) {
        // Each hidden layer has a number of inputs equal to the previous
        // layer's number of neurons
        layers.push_back(Layer(neurons_per_layer[i-1], neurons_per_layer[i],
                                &Sigmoid, &SigmoidDerivative));
    }

    // Output layer
    layers.push_back(Layer(neurons_per_layer.back(), num_outputs, 
                           &Sigmoid, &SigmoidDerivative));
}

Layer::Layer(const int& num_input_nodes, const int& num_neurons,
             double (*ActivationFunction)(double),
             double (*ActivationFunctionDerivative)(double)) :
             num_inputs(num_input_nodes) {
    for (int i = 0; i < num_neurons; i++) {
        neurons.push_back(Neuron(num_input_nodes, ActivationFunction,
                                 ActivationFunctionDerivative));
    }
}

Neuron::Neuron(const int& num_input_nodes, double (*ActivationFunction)(double),
               double (*ActivationFunctionDerivative)(double)) :
               ActivationFunction_(ActivationFunction),
               ActivationFunctionDerivative_(ActivationFunctionDerivative) {
    bias = RandRange(-1, 1);
    for (int i = 0; i < num_input_nodes; i++) {
        weights.push_back(RandRange(-1, 1));
    }
}

// =======================================
// Forward Propagation Methods
// =======================================

std::vector<double> NeuralNetwork::Forwards(const std::vector<double>& input) {
    if (num_inputs_ != input.size()) {
        throw std::runtime_error("Input size mismatch in NeuralNetwork::Forward"
                    "s. Input size is " + std::to_string(input.size()) 
                    + ", expected input size is " + std::to_string(num_inputs_));
    }

    std::vector<double> next_input = input;
    std::vector<double> current_output;

    for (Layer& layer : layers) {
        current_output = layer.Forwards(next_input);
        next_input = current_output;
    }

    return last_output = current_output;
}

std::vector<double> Layer::Forwards(const std::vector<double>& inputs) {
    if (num_inputs != inputs.size()) {
        throw std::runtime_error("Input size mismatch in Layer::Forwards. "
                    "Input size is " + std::to_string(inputs.size()) 
                    + ", expected input size is " + std::to_string(num_inputs));
    }

    std::vector<double> output;
    for (Neuron& neuron : neurons) {
        output.push_back(neuron.Forwards(inputs));
    }

    return output;
}

double Neuron::Forwards(const std::vector<double>& inputs) {
    if (inputs.size() != weights.size()) {
            throw std::runtime_error("Input size mismatch in Neuron::Forwards. "
                        "Input size is " + std::to_string(inputs.size()) 
                        + ", weight size is " + std::to_string(weights.size()));
    }

    latest_input = inputs;

    double result = bias;
    for (int i = 0; i < inputs.size(); i++) {
        result += inputs.at(i) * weights.at(i);
    }

    latest_output = ActivationFunction_(result);

    return latest_output;
}

// =======================================
// Backward Propagation Methods
// =======================================

void NeuralNetwork::Backwards(const std::vector<double>& target) {
    if (num_outputs_ != target.size()) {
        throw std::runtime_error("Input size mismatch in NeuralNetwork::Backwar"
                    "ds. Target size is " + std::to_string(target.size()) 
                    + ", expected target size is " + std::to_string(num_outputs_));
    }

    std::vector<std::vector<double>> dCost_dOutput;
    dCost_dOutput.push_back(Calculate_dCostdOutput(target));
    std::vector<std::vector<double>> dCost_dInput;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        dCost_dInput = it->Backwards(dCost_dOutput);
        dCost_dOutput = dCost_dInput;
    }
}

// TODO: instead of returning the whole vector, process a running sum
//       of the mean error for each neuron on the previous layer
std::vector<std::vector<double>> Layer::Backwards(const std::vector<std::vector
                                                  <double>>& dCost_dOutput) {
    for (const auto& single_dCost : dCost_dOutput) {
        if (neurons.size() != single_dCost.size()) {
                throw std::runtime_error("Input size mismatch in Layer::Backwar"
                "ds. dCost_dOutput is " + std::to_string(single_dCost.size()) 
                + ", number of neurons is " + std::to_string(neurons.size()));
        }
    }

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

std::vector<double> Neuron::Backwards(const double& mean_dCost_dOutpuy) {
    if (latest_input.size() != weights.size()) {
            throw std::runtime_error("Input size mismatch in Neuron::Backwards."
                        " Input size is " + std::to_string(latest_input.size()) 
                        + ", weight size is " + std::to_string(weights.size()));
    }

    double delta = mean_dCost_dOutpuy 
                   * ActivationFunctionDerivative_(latest_output);

    // Bias change: -(learning rate * error * activation function derivative)
    bias -= LEARNING_RATE * delta;

    // Weight change: -(learning rate * error *
    //           activation function derivative * output of previous layer)
    for (int i = 0; i < weights.size(); i++) {
        weights.at(i) -= LEARNING_RATE * latest_input.at(i) * delta;
    }

    // Cost to previous layer: -(learning rate * error *
    //           activation function derivative * weight)
    std::vector<double> dCost_dInput;
    for (const double& weight : weights) {
        dCost_dInput.push_back(weight * delta);
    }
    
    return dCost_dInput;
}

// =======================================
// Network Interface Methods
// =======================================

std::vector<double> NeuralNetwork::CalculateError(const std::vector<double>&
                                                  target) {
    if (last_output.size() != target.size()) {
        throw std::runtime_error("Input size mismatch in NeuralNetwork::Calcula"
            "teError. Target size is " + std::to_string(target.size()) 
            + ", last output size is " + std::to_string(last_output.size()));
    }

    std::vector<double> error;
    for (int i = 0; i < last_output.size(); i++) {
        // Mean squared error
        error.push_back(pow(last_output.at(i) - target.at(i), 2));
    }

    return error;
}

std::vector<double> NeuralNetwork::Calculate_dCostdOutput(const std::vector
                                                          <double>& target) {
    if (last_output.size() != target.size()) {
        throw std::runtime_error("Input size mismatch in NeuralNetwork::Calcula"
            "te_dCostdOutput. Target size is " + std::to_string(target.size()) 
            + ", last output size is " + std::to_string(last_output.size()));
    }

    std::vector<double> dCost_dOutput;
    for (int i = 0; i < last_output.size(); i++) {
        // Mean squared error derivative
        dCost_dOutput.push_back(2 * (last_output.at(i) - target.at(i)));
    }

    return dCost_dOutput;
}

// =======================================
// Utility and Debug Methods
// =======================================

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

void Layer::PrintLayer() const {
    for (int i = 0; i < neurons.size(); i++) {
        printf("Neuron %d: ", i);
        neurons.at(i).PrintNeuron();
        printf("\t");
    }
}

const void Neuron::PrintNeuron() const {
    printf("(o = %.2f; b = %.2f", latest_output, bias);
    for (int i = 0; i < weights.size(); i++) {
        printf("; w_%d = %.2f", i, weights.at(i));
    }
    printf(")");
}