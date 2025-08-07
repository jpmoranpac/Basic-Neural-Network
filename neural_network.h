#include <vector>
#include <algorithm>
#include <random>

class Neuron {
public:
    double bias = 0.0;
    std::vector<double> weights;
    double (*ActivationFunction_)(double);
    double (*ActivationFunctionDerivative_)(double);

    double latest_output;
    Neuron(const int& num_input_nodes,
           double (*ActivationFunction)(double),
           double (*ActivationFunctionDerivative)(double));

    double Forwards(const std::vector<double> inputs);

    std::vector<double> Backwards(const double& mean_dCost_dOutpuy);

    const void PrintNeuron() const;
};

class Layer {
public:
    std::vector<Neuron> neurons;
    int num_inputs = 0;
    
    Layer(const int& num_input_nodes, const int& num_neurons,
          double (*ActivationFunction)(double),
          double (*ActivationFunctionDerivative)(double));

    std::vector<double> Forwards(const std::vector<double> inputs);

    std::vector<std::vector<double>> Backwards(
                        const std::vector<std::vector<double>>& dCost_dOutput);

    void PrintLayer() const;
};

class NeuralNetwork {
public:
    std::vector<Layer> layers;
    std::vector<double> last_output;
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    
    NeuralNetwork(const int& num_inputs, const int& num_outputs, 
                  const std::vector<int>& neurons_per_layer);

    std::vector<double> Forwards(const std::vector<double>& input);

    void Backwards(const std::vector<double>& target);

    std::vector<double> CalculateError(const std::vector<double>& target);

    std::vector<double> Calculate_dCostdOutput(
                                            const std::vector<double>& target);

    void PrintNetwork() const;
};