#include <iostream>

#include "load_data.h"
#include "neural_network.h"
#include "tank_counting.h"
#include "neural_network_demo.h"

int TankTraining() {
    // Frequentist sample output:
    {
    for (int i = 0; i < 10; i++) {
        TankPopulationExercise ex = CreateTankPopulationExercise(100, 1000, 15);
        const int pop = ex.true_population;
        const int pred = FrequentistPrediction(ex.population_peeks);
        const double error = abs(pred - pop) / static_cast<double>(pop);
        printf("Tank population: %4.0d, Prediction: %4.0d, Error: %.2f%%\n",
                pop, pred, error * 100);
    }

    const int kTotalRuns = 10000;
    double mean_error = 0.0;
    for (int i = 0; i < kTotalRuns; i++) {
        TankPopulationExercise ex = CreateTankPopulationExercise(20, 1000, 20);
        const int pop = ex.true_population;
        const int pred = FrequentistPrediction(ex.population_peeks);
        mean_error += abs(pred - pop) / static_cast<double>(pop) / kTotalRuns;
    }
    printf("Mean error over %d runs: %.2f%%\n",
            kTotalRuns, mean_error * 100);
    }

    // NN solution:
    {
    const int kEpoch = 50;
    const int kBatchSize = 1000;
    const int kTankMin = 100;
    const int kTankMax = 1000;
    const int kTankPeeks = 15;

    std::vector<int> hidden_layers = {100, 100};
    NeuralNetwork network = NeuralNetwork(kTankPeeks, 1, hidden_layers);

    printf("Beginning training...\n");

    for (int epoch = 0; epoch < kEpoch; epoch++) {
        double success_count = 0.0;
        double mean_loss = 0.0;

        for (int i = 0; i < kBatchSize; i++) {
            auto ex = CreateTankPopulationExercise(kTankMin, kTankMax, kTankPeeks);

            // Convert population peeks from ints to a percentage of the max pop
            std::vector<double> input;
            for (auto& p : ex.population_peeks) {
                input.emplace_back(static_cast<double>(p)/kTankMax);
            }

            // Same for population count
            const double pop = static_cast<double>(ex.true_population) / 
                               static_cast<double>(kTankMax);
            std::vector<double> target;
            target.emplace_back(pop);
            
            // Forward propagation
            std::vector<double> output;
            output = network.Forwards(input);
            
            // Backwards propagation, including update weights and biases
            network.Backwards(target);

            // Keep track of the number of succsseful predictions
            double prediction = output.at(0) * kTankMax;
            success_count += (abs(prediction - pop) < 0.1);
            
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
        auto ex = CreateTankPopulationExercise(kTankMin, kTankMax, kTankPeeks);

        // Convert population peeks from ints to a percentage of the max pop
        std::vector<double> input;
        for (auto& p : ex.population_peeks) {
            input.emplace_back(static_cast<double>(p)/kTankMax);
        }
        
        // Forward propagation
        std::vector<double> output;
        output = network.Forwards(input);

        // Keep track of the number of succsseful predictions
        double prediction = output.at(0) * kTankMax;
        const double error = abs(prediction - ex.true_population) /
                             static_cast<double>(ex.true_population);
        printf("Tank population: %4.0d, Prediction: %4.0f, Error: %.2f%%\n",
                ex.true_population, prediction, error * 100);
    }

    const int kTotalRuns = 10000;
    double mean_error = 0.0;
    for (int i = 0; i < kTotalRuns; i++) {
        auto ex = CreateTankPopulationExercise(kTankMin, kTankMax, kTankPeeks);

        // Convert population peeks from ints to a percentage of the max pop
        std::vector<double> input;
        for (auto& p : ex.population_peeks) {
            input.emplace_back(static_cast<double>(p)/kTankMax);
        }
        
        // Forward propagation
        std::vector<double> output;
        output = network.Forwards(input);

        // Keep track of the number of succsseful predictions
        double prediction = output.at(0) * kTankMax;
        const double error = abs(prediction - ex.true_population) /
                             static_cast<double>(ex.true_population);

        mean_error += error / kTotalRuns;
    }
    printf("Mean error over %d runs: %.2f%%\n",
            kTotalRuns, mean_error * 100);
    }
    
    return 0;
}

int main(int argc, char** argv) {
    int seed = time(NULL);
    printf("Seed: %d\n", seed);
    srand(seed);

    MnistExample();
}