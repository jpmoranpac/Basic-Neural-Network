#include <random>
#include <algorithm>
#include <vector>

/// @brief Stores a number of observations of the serial number population 
///        (population_peeks) and the true population count (true_population).
///        Does not store all serial numbers.
struct TankPopulationExercise {
    const int true_population;
    const std::vector<int> population_peeks;
};

/// @brief Returns a shuffled list of serial numbers starting from one and
///        linearly increasing to a random number between min and max.
/// @param min the lowest population size
/// @param max the highest population size
/// @return shuffled serial number population
std::vector<int> GenerateRandomTankPopulation(const int& min, const int& max);

/// @brief Return the first x elements of the population, where x is the number 
///        of peeks requested. Will not return duplicate peeks of the same
///        serial number. Assumed population is pre-shuffled.
/// @param tank_population a pre-shuffled population of tank serial numbers
/// @param number_of_peeks number of peeks to perform
/// @return a vector of integers representing the serial numbers peeked
std::vector<int> PeekTankSerialNumbers(const std::vector<int>& tank_population,
                                       const int& number_of_peeks);

/// @brief Generates a set of observations for a set of serial numbers.
/// @param min_population the lowest population size
/// @param max_population the highest population size
/// @param number_of_peeks number of observations to generate
/// @return struct containing the set of observations and true population size
TankPopulationExercise CreateTankPopulationExercise(const int& min_population,
                                                    const int& max_population,
                                                    const int& number_of_peeks);

/// @brief Calculates the "Frequentist" solution to the German tank counting
///        problem. Specifically, N = m + m/k - 1, where m is the highest seen
///        serial number, k is the number of observations, and N is the
///        estimated population size.
/// @param tank_population list of serial number observations
/// @return estimated population size
int FrequentistPrediction(const std::vector<int> tank_population);