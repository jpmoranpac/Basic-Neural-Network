#include <unordered_map>
#include <string>
#include <variant>
#include <vector>
#include <sstream>

// Define ConfigType as a variant that holds pointers to data
using ConfigValueType = std::variant<int*, double*, std::string*, 
                                     std::vector<int>*>;

// Struct used to map from the name of a config item to local storage
struct FieldMapping {
    std::string key;
    ConfigValueType target;
};

/// @brief Handles loading INI config files and reading its contents into an
///        arbitrary struct.
///        Compatible with the types int, double, string, and vector<int>.
///        Example usage:
///
///    // Load the INI config file
///    static auto config = Config("config.ini");
///
///    // Define the struct to contain the config data
///    struct {
///        int epochs = 0;
///        double learning_rate = 0.0;
///        std::string activation = "";
///        std::vector<int> hidden_layers;
///    } general_cfg;
///
///    // Set the struct's data fields from the loaded INI file
///    config.LoadStructFromConfig(general_cfg, {
///        {"epochs", &general_cfg.epochs},
///        {"learning_rate", &general_cfg.learning_rate},
///        {"activation", &general_cfg.activation},
///        {"hidden_layers", &general_cfg.hidden_layers},
///    });
class Config {
private:
    /// Contains the loaded INI data
    std::unordered_map<std::string, std::string> values;

    /// @brief Returns config data as the specified type. 
    ///        Throws runtime_error if the key doesn't exist.
    /// @tparam T The type to return the data
    /// @param key string key of the data
    /// @return loaded data
    template<typename T> const T Get(const std::string& key) const;

    /// @brief Returns config data as a list of the specified type. 
    ///        Throws runtime_error if the key doesn't exist.
    /// @tparam T The type to return the data
    /// @param key string key of the data
    /// @return loaded data
    template<typename T> std::vector<T> GetList(const std::string& key,
                                                char delimiter=',') const;

public:
    /// @brief Initialiser. Must pass the path to an INI config file.
    ///        Throws runtime_error if the file cannot be read
    /// @param filename INI file to read
    Config(const std::string& filename);

    /// @brief Populates an arbitrary struct with data loaded from the INI
    ///        config file. Supports int, double, string, and vector<int>
    /// @tparam Struct arbitrary struct to populate
    /// @param s arbitrary struct to populate
    /// @param fields mapping from keys to struct fields
    template<typename Struct> void LoadStructFromConfig(Struct& s,
                                            std::vector<FieldMapping> fields);
};

template<typename Struct>
void Config::LoadStructFromConfig(Struct& s, std::vector<FieldMapping> fields) {
    for (auto& f : fields) {
        std::visit([&](auto* memberPtr) {
            using T = std::decay_t<decltype(*memberPtr)>;
            if constexpr (std::is_same_v<T, std::vector<int>>) {
                *memberPtr = GetList<int>(f.key);
            } else {
                *memberPtr = Get<T>(f.key);
            }
        }, f.target);
    }
}

template<typename T> const T Config::Get(const std::string& key) const {
    auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("Missing expected key: " + key);
    }
    
    std::istringstream iss(it->second);
    T result;
    iss >> result;
    return result;
}

template<typename T> std::vector<T> Config::GetList(const std::string& key,
                                            char delimiter) const {
    auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("Missing list config key: " + key);
    }

    std::vector<T> result;
    std::istringstream ss(it->second);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        std::istringstream iss(item);
        T value; iss >> value;
        result.push_back(value);
    }
    return result;
}
