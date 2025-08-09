#include <iostream>

#include "load_data.h"
#include "neural_network.h"

int SimpleExample() {
    // Create NN: input=2, one hidden layer with 2 neurons, output=1 neuron
    std::vector<int> hidden_layers = {2};
    NeuralNetwork nn(2, 1, hidden_layers);

    // Sample input and target output
    std::vector<double> input = {0.5, -0.3};
    std::vector<double> target = {0.7};

    for (int epoch = 0; epoch <= 1000; ++epoch) {
        auto output = nn.Forwards(input);
        nn.Backwards(target);
        if ((epoch) % 100 == 0) {
            double loss = nn.CalculateError(target).at(0);
            printf("Epoch %d loss %f\n", epoch, loss);
        }
    }

    return 0;
}

int MnistExample() {
    printf("Loading data...\n");
    std::vector<std::vector<double>> images_train;
    std::vector<int> labels_train;
    std::vector<std::vector<double>> images_test;
    std::vector<int> labels_test;
    bool loaded = LoadData(images_train, labels_train,
                            images_test, labels_test);
    if (!loaded)
    {
        printf("Failed to load data.\n");
        return 1;
    }

    std::vector<int> hidden_layers = {100, 100};
    NeuralNetwork network = NeuralNetwork(28 * 28, 10, hidden_layers);

    const int kEpoch = 50;
    const int kBatchSize = 100;

    printf("Beginning training...\n");

    for (int epoch = 0; epoch < kEpoch; epoch++) {
        // Selecte a random batch of training sample
        std::vector<int> training_indices;
        for (int j = 0; j < kBatchSize; j++) {
            training_indices.push_back(rand() % images_train.size());
        }

        double success_count = 0.0;
        double mean_loss = 0.0;

        for (const int& sample : training_indices) {
            // Training is labelled with a single number rather
            // than a vector, so create the target vector here
            const int label = labels_train.at(sample);
            std::vector<double> target;
            target.resize(10, 0.0);
            target.at(label) = 1.0;

            // Forward propagation
            std::vector<double> output;
            output = network.Forwards(images_train.at(sample));
            
            // Backwards propagation, including update weights and biases
            network.Backwards(target);

            // Keep track of the number of succsseful predictions
            int prediction = 0;
            for (int j = 0; j < output.size(); j++)
            {
                if (output[j] > output[prediction])
                {
                    prediction = j;
                }
            }
            success_count += prediction == label;
            
            std::vector<double> loss = network.CalculateError(target);
            auto const count = static_cast<float>(loss.size());
            mean_loss += std::reduce(loss.begin(), loss.end()) / count
                         / kBatchSize;
        }

        double success_rate = success_count / kBatchSize;

        printf("Epoch %d success rate: %.0f%% mean loss: %f\n",
                epoch, success_rate*100, mean_loss);
    }

    // Print a selection of random images to demonstrate learning
    for (int i = 0; i < 10; i++) {
        int index = rand() % images_train.size();
        std::vector<double> image = images_train[index];
        int label = labels_train[index];

        std::vector<double> output = network.Forwards(image);

        int prediction = 0;
        for (int j = 0; j < output.size(); j++)
        {
            if (output[j] > output[prediction])
            {
                prediction = j;
            }
        }

        PrintAsciiImage(image);
        printf("Label is: %d, Predicted: %d\n", label, prediction);
    }

    return 0;
}

int main(int argc, char** argv) {
    int seed = time(NULL);
    printf("Seed: %d\n", seed);
    srand(seed);

    return MnistExample();
}