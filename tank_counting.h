#include <random>
#include <algorithm>
#include <vector>

struct TankPopulationExercise {
    const int true_population;
    const std::vector<int> population_peeks;
};

std::vector<int> GenerateRandomTankPopulation(const int& min, const int& max);

/// @brief Return the first x elements of the population, where x is the number 
///        of peeks requested. Will not return duplicate peeks of the same
///        serial number. Assumed population is pre-shuffled.
/// @param tank_population a pre-shuffled population of tank serial numbers
/// @param number_of_peeks number of peeks to perform
/// @return a vector of integers representing the serial numbers peeked
std::vector<int> PeekTankSerialNumbers(const std::vector<int>& tank_population,
                                       const int& number_of_peeks);

TankPopulationExercise CreateTankPopulationExercise(const int& min_population,
                                                    const int& max_population,
                                                    const int& number_of_peeks);

int FrequentistPrediction(const std::vector<int> tank_population);