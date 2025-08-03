/*
    Heavily based on the work https://github.com/Krish120003/CPP_Neural_Network
*/

#include <vector>
#include <iostream>
#include <fstream>

void ReverseBytes(char *bytes, const int& size);

bool ReadBigEndian(std::ifstream& file, char *bytes, const uint& size);

bool LoadLabelDatabaseFile(const std::string& filename,
                          std::vector<int>& output);

bool LoadImageDatabaseFile(const std::string& filename,
                          std::vector<std::vector<double>>& output);

bool LoadData(std::vector<std::vector<double>>& images_train,
               std::vector<int>& labels_train,
               std::vector<std::vector<double>>& images_test,
               std::vector<int>& labels_test);

void PrintAsciiImage(const std::vector<double> &values);