/*
    Heavily based on the work https://github.com/Krish120003/CPP_Neural_Network
*/

#include <vector>
#include <iostream>
#include <fstream>

/// @brief Reverses byte order as a conversion between big endian and little
///        endian.
/// @param bytes pointer to the data
/// @param size number of bytes in the data
void ReverseBytes(char *bytes, const int& size);

/// @brief Read a file stream encoded in big endian for a little endian system
/// @param file file stream to read
/// @param bytes pointer to buffer to store data
/// @param size number of bytes read
/// @return success
bool ReadBigEndian(std::ifstream& file, char *bytes, const uint& size);

/// @brief Loads an MNIST label database - either training or test data. Returns
///        a vector of ints, where each element is a number from 0 - 9.
/// @param filename full filepath to the MNIST label file
/// @param output vector of data labels
/// @return success
bool LoadLabelDatabaseFile(const std::string& filename,
                          std::vector<int>& output);

/// @brief Loads an MNIST image database - either training or test data. Returns
///        data as a "2D" vector array. The outer vector is the index of 
///        image. The inner vector is a 784 bytes long vector, where each 
///        element is a double from 0.0-1.0 representing the intensity of that
///        pixel. That is, each image is deconstucted into a 1D vector.
/// @param filename full filepath to the MNIST image file
/// @param output nested vector of image data
/// @return success
bool LoadImageDatabaseFile(const std::string& filename,
                          std::vector<std::vector<double>>& output);

/// @brief Loads a set of MNIST image and label databases for both training and
///        test data. Labels are a vector of ints, where each element is a
///        number from 0 - 9. Image data is a "2D" vector array. The outer
///        vector is the index of image. The inner vector is a 784 bytes long
///        vector, where each element is a double from 0.0-1.0 representing the
///        intensity of that pixel. That is, each image is deconstucted into a
///        1D vector.
/// @param images_train output vector to write training images
/// @param labels_train output vector to write training labels
/// @param images_test output vector to write test images
/// @param labels_test output vector to write test labels
/// @return success
bool LoadData(std::vector<std::vector<double>>& images_train,
               std::vector<int>& labels_train,
               std::vector<std::vector<double>>& images_test,
               std::vector<int>& labels_test);

/// @brief Prints to the terminal an ASCII interpretation of a 1D MNIST image
///        file. The file is assumed to be a 28x28 px image.
/// @param values the image to print as a 1D array
void PrintAsciiImage(const std::vector<double> &values);