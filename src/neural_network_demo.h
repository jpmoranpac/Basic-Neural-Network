/// @brief Demo training a neural network (2x2x1) on a static training data
///        point.
void SimpleExample(const int& epochs, const std::vector<int>& hidden_layers);

/// @brief Loads the mnist dataset and trains a neural network (784x100x100x10)
///        to identify hand written digits. Prints epoch results and examples
///        from the test dataset.
void MnistExample(const int& epochs, const int& batch_size,
                 const int& test_count, const std::vector<int>& hidden_layers);