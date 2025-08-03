#include <vector>
#include <iostream>
#include <fstream>

void ReverseBytes(char *bytes, const int& size) {
    for (int i = 0; i < size / 2; i++)
    {
        char temp = bytes[i];
        bytes[i] = bytes[size - i - 1];
        bytes[size - i - 1] = temp;
    }
}

bool ReadBigEndian(std::ifstream& file, char *bytes, const uint& size) {
    file.read(bytes, size);
    ReverseBytes(bytes, size);

    return true;
}

bool LoadLabelDatabaseFile(const std::string& filename,
                          std::vector<int>& output) {
    std::ifstream labels_file;
    labels_file.open(filename, std::ios::binary | std::ios::in);
    if (!labels_file.is_open())
    {
        return false;
    }

    labels_file.seekg(0, std::ios::beg);

    // First 32 bits are magic number
    int magic_number;
    ReadBigEndian(labels_file, reinterpret_cast<char*>(&magic_number),
                    sizeof(magic_number));

    if (magic_number != 2049) {
        printf("ERROR: \"%s\" is not an mnist label database. "
               "Magic number is: %d\n", filename.c_str(), magic_number);
        return false;
    }

    // Next 32 bits are number of items
    int number_of_items;
    ReadBigEndian(labels_file, reinterpret_cast<char*>(&number_of_items),
                    sizeof(number_of_items));

    // Read labels
    for (int i = 0; i < number_of_items; i++)
    {
        // Each label is 1 byte, so we use a char
        char label;
        ReadBigEndian(labels_file, &label, sizeof(label));
        output.push_back((int)label);
    }

    labels_file.close();

    return true;
}

bool LoadImageDatabaseFile(const std::string& filename,
                          std::vector<std::vector<double>>& output) {
    std::ifstream images_file;
    images_file.open(filename, std::ios::binary | std::ios::in);
    if (!images_file.is_open())
    {
        return false;
    }

    // Seek to beginning of file
    images_file.seekg(0, std::ios::beg);

    // First 32 bits are magic number
    int magic_number;
    ReadBigEndian(images_file, reinterpret_cast<char*>(&magic_number),
                    sizeof(magic_number));

    if (magic_number != 2051) {
        printf("ERROR: \"%s\" is not an mnist label database. "
               "Magic number is: %d\n", filename.c_str(), magic_number);
        return false;
    }

    // Next 32 bits are number of items
    int number_of_items;
    ReadBigEndian(images_file, reinterpret_cast<char*>(&number_of_items),
                    sizeof(number_of_items));

    // Read number of rows
    int number_of_rows;
    ReadBigEndian(images_file, reinterpret_cast<char*>(&number_of_rows),
                    sizeof(number_of_rows));

    // Read number of columns
    int number_of_columns;
    ReadBigEndian(images_file, reinterpret_cast<char*>(&number_of_columns),
                    sizeof(number_of_columns));

    // Read images_train
    for (int i = 0; i < number_of_items; i++)
    {
        // Each image is 28 * 28 = 784 bytes
        char image[784];
        // No need to read big-endian - the image is stored as single characters
        // read left-to-right, top-to-bottom
        images_file.read(image, sizeof(image));

        // Convert to std::vector of doubles
        std::vector<double> image_vector;
        for (int j = 0; j < 784; j++)
        {
            unsigned int temp = (unsigned int)((unsigned char)image[j]);
            // We normalize the values to be between 0 and 1
            // By dividing by 255, the maximum value of a byte
            image_vector.push_back((double)(temp) / 255.0);
        }

        output.push_back(image_vector);
    };

    images_file.close();

    return true;
}

bool LoadData(std::vector<std::vector<double>>& images_train,
               std::vector<int>& labels_train,
               std::vector<std::vector<double>>& images_test,
               std::vector<int>& labels_test) {
    bool success = true;
    success &= LoadLabelDatabaseFile("data/train-labels-idx1-ubyte",
                                    labels_train);
    success &= LoadImageDatabaseFile("data/train-images-idx3-ubyte",
                                    images_train);
    success &= LoadLabelDatabaseFile("data/t10k-labels-idx1-ubyte",
                                    labels_test);
    success &= LoadImageDatabaseFile("data/t10k-images-idx3-ubyte",
                                    images_test);

    return success;
}

void PrintAsciiImage(const std::vector<double> &values) {
    // Check if the size of the input std::vector is correct
    std::cout << "values.size(): " << values.size() << std::endl;

    if (values.size() != 784)
    {
        std::cerr << "Error: Input std::vector size is not 784." << std::endl;
        return;
    }

    // Define characters to represent different intensity levels
    const char intensityChars[] = {' ', '.', ',', ':', 'o',
                                   'O', 'X', '#', '$', '@'};

    // Calculate the range for each intensity level
    const double range = 1.0 / (sizeof(intensityChars)
                         / sizeof(intensityChars[0]) - 1);

    // Iterate over the std::vector and print ASCII characters based on values
    for (int i = 0; i < values.size(); ++i)
    {
        // Adjust the intensity to a character in the range of ASCII characters
        int intensityLevel = static_cast<int>(values[i] / range);
        intensityLevel = std::min(std::max(intensityLevel, 0),
                         static_cast<int>(sizeof(intensityChars) - 1));

        char pixel = intensityChars[intensityLevel];

        // Print the ASCII character
        std::cout << pixel;

        // Insert a newline character after every
        // 28 characters to create a 28x28 image
        if ((i + 1) % 28 == 0)
        {
            std::cout << std::endl;
        }
    }
}
