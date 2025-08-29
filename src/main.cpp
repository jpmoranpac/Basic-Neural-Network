#include <iostream>

#include "load_data.h"
#include "neural_network.h"
#include "tank_counting.h"
#include "neural_network_demo.h"
#include "config.h"

int TankTraining(const int& epochs, const int& batch_size, const int& tank_min,
                 const int& tank_max, const int& tank_peeks,
                 const int& test_count, const std::vector<int>& hidden_layers) {
    // Frequentist sample output:
    double mean_error = 0.0;
    for (int i = 0; i < test_count; i++) {
        TankPopulationExercise ex =
                CreateTankPopulationExercise(tank_min, tank_max, tank_peeks);
        const int pop = ex.true_population;
        const int pred = FrequentistPrediction(ex.population_peeks);
        mean_error += abs(pred - pop) / static_cast<double>(pop) / test_count;
    }
    printf("Mean error over %d runs: %.2f%%\n",
            test_count, mean_error * 100);

    // NN solution:
    NeuralNetwork network = NeuralNetwork(tank_peeks, 1, hidden_layers);

    printf("Beginning training...\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        double success_count = 0.0;
        double mean_loss = 0.0;

        for (int i = 0; i < batch_size; i++) {
        TankPopulationExercise ex =
                CreateTankPopulationExercise(tank_min, tank_max, tank_peeks);

            // Convert population peeks from ints to a percentage of the max pop
            std::vector<double> input;
            for (auto& p : ex.population_peeks) {
                input.emplace_back(static_cast<double>(p)/tank_max);
            }

            // Same for population count
            const double pop = static_cast<double>(ex.true_population) / 
                               static_cast<double>(tank_max);
            std::vector<double> target;
            target.emplace_back(pop);
            
            // Forward propagation
            std::vector<double> output;
            output = network.Forwards(input);
            
            // Backwards propagation, including update weights and biases
            network.Backwards(target);

            // Keep track of the number of succsseful predictions
            double prediction = output.at(0) * tank_max;
            success_count += (abs(prediction - pop) < 0.1);
            
            std::vector<double> loss = network.CalculateError(target);
            auto const count = static_cast<float>(loss.size());
            mean_loss += std::reduce(loss.begin(), loss.end()) / count
                         / tank_max;
        }

        double success_rate = success_count / tank_max;

        printf("Epoch %d success rate: %.0f%% mean loss: %f\n",
                epoch, success_rate*100, mean_loss);
    }

    const int kTotalRuns = 10000;
    mean_error = 0.0;
    for (int i = 0; i < kTotalRuns; i++) {
        TankPopulationExercise ex =
                CreateTankPopulationExercise(tank_min, tank_max, tank_peeks);

        // Convert population peeks from ints to a percentage of the max pop
        std::vector<double> input;
        for (auto& p : ex.population_peeks) {
            input.emplace_back(static_cast<double>(p)/tank_max);
        }
        
        // Forward propagation
        std::vector<double> output;
        output = network.Forwards(input);

        // Keep track of the number of succsseful predictions
        double prediction = output.at(0) * tank_max;
        const double error = abs(prediction - ex.true_population) /
                             static_cast<double>(ex.true_population);

        mean_error += error / kTotalRuns;
    }
    printf("Mean error over %d runs: %.2f%%\n",
            kTotalRuns, mean_error * 100);
    
    return 0;
}

int main(int argc, char** argv) {
    int seed = time(NULL);
    printf("Seed: %d\n", seed);
    srand(seed);

    static auto config = Config("config.ini");

    struct {
        int epochs = 0;
        int batch_size = 0;
        int test_count = 0;
        double learning_rate = 0.0;
        std::string activation = "";
        std::vector<int> hidden_layers;
        std::string demo = "";
    } general_cfg;

    config.LoadStructFromConfig(general_cfg, {
        {"epochs", &general_cfg.epochs},
        {"batch_size", &general_cfg.batch_size},
        {"test_count", &general_cfg.test_count},
        {"learning_rate", &general_cfg.learning_rate},
        {"activation", &general_cfg.activation},
        {"hidden_layers", &general_cfg.hidden_layers},
        {"demo", &general_cfg.demo}
    });

    if (general_cfg.demo == "tank") {
        struct {
            int tank_min = 0;
            int tank_max = 0;
            int tank_peeks = 0;
        } tank_cfg;

        config.LoadStructFromConfig(tank_cfg, {
            {"tank_min", &tank_cfg.tank_min},
            {"tank_max", &tank_cfg.tank_max},
            {"tank_peeks", &tank_cfg.tank_peeks},
        });

        TankTraining(general_cfg.epochs, general_cfg.batch_size, 
                     tank_cfg.tank_min, tank_cfg.tank_max, tank_cfg.tank_peeks,
                     general_cfg.test_count, general_cfg.hidden_layers);
    }
    else if (general_cfg.demo == "mnist") {
        MnistExample(general_cfg.epochs, general_cfg.batch_size,
                     general_cfg.test_count, general_cfg.hidden_layers);
    }
    else if (general_cfg.demo == "simple") {
        SimpleExample(general_cfg.epochs, general_cfg.hidden_layers);
    }
    else {
        printf("Unknown demo type \"%s\"\n", general_cfg.demo.c_str());
    }

    return 0;
}