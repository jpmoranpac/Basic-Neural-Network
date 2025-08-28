#include <fstream>

#include "config.h"

Config::Config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open config file: " + filename);

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;

        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            values[key] = value;
        }
    }
}
