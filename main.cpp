#include <iostream>

#include "load_data.h"
#include "neural_network.h"

int main(int argc, char** argv) {
    int seed = time(NULL);
    printf("Seed: %d\n", seed);
    srand(seed);

    std::vector<int> hidden_layers = {2, 2};
    NeuralNetwork(3, 3, hidden_layers);

    #if 0
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

    // Print a selection of random images to verify load was successful
    for (int i = 0; i < 5; i++) {
        int index = rand() % images_test.size();
        std::vector<double> image = images_test[index];
        int label = labels_test[index];

        PrintAsciiImage(image);
        std::cout << "Random Test Image and Label: " << label << std::endl;

        index = rand() % images_train.size();
        image = images_train[index];
        label = labels_train[index];

        PrintAsciiImage(image);
        std::cout << "Random Train Image and Label: " << label << std::endl;
    }
    #endif

    return 0;
}