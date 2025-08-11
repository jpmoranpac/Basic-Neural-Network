#include <random>
#include <algorithm>
#include <vector>

auto random_source = std::mt19937{std::random_device{}()};

struct TankPopulationExercise {
    const int true_population;
    const std::vector<int> population_peeks;
};

std::vector<int> GenerateRandomTankPopulation(const int& min, const int& max) {
    // Decide the total number of tanks
    std::uniform_int_distribution<> dist(min, max);
    const int population_count = dist(random_source);

    // Create the population with serially increasing assigned numbers
    // TODO: Extension - what if some tanks get destroyed early?
    // TODO: Extension - what if the serial number doesn't start at zero?
    // TODO: Extension - what if the serial numbers are encoded? With a private
    //                   key?

    std::vector<int> tank_population(population_count);
    std::iota(tank_population.begin(), tank_population.end(), 1);

    // Pre-shuffle the population
    std::shuffle(tank_population.begin(), tank_population.end(), random_source);

    return tank_population;
}

std::vector<int> PeekTankSerialNumbers(const std::vector<int>& tank_population,
                                       const int& number_of_peeks) { 
    return std::vector<int>(tank_population.begin(),
                            tank_population.begin() + number_of_peeks);
}

TankPopulationExercise CreateTankPopulationExercise(const int& min_population,
                                                    const int& max_population,
                                                    const int& number_of_peeks) {
    auto tank_population = GenerateRandomTankPopulation(min_population,
                                                        max_population);
    auto population_peeks = PeekTankSerialNumbers(tank_population,
                                                  number_of_peeks);

    TankPopulationExercise x = {static_cast<int>(tank_population.size()),
                                population_peeks};

    return x;
}

int FrequentistPrediction(const std::vector<int> tank_population) {
    // m: highest number seen
    // k: number of observations
    const int m = *(std::max_element(tank_population.begin(),
                                     tank_population.end()));
    const int k = static_cast<int>(tank_population.size());
    
    return m + m / k - 1;
}