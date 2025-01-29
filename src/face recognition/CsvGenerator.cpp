#include "CsvGenerator.h"
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

bool generateCsv(const std::string& directory, const std::string& csvFileName)
{
    try
    {
        std::ofstream csvFile(csvFileName);
        if (!csvFile.is_open())
        {
            std::cerr << "Failed to open file: " << csvFileName << std::endl;
            return false;
        }



        int label = 0; // 自增标签

        // 遍历目录中的文件
        for (const auto& entry : fs::directory_iterator(directory))
        {
            if (entry.is_regular_file() && (entry.path().extension() == ".jpg"|| entry.path().extension() == ".png"|| entry.path().extension() == ".pgm"|| entry.path().extension() == ".webp")) // 确保是图像文件
            {
                // 写入图像路径和自增标签
                csvFile << entry.path().string() << ";" << label++ << "\n";
            }
        }

        csvFile.close();
        std::cout << "CSV file generated successfully: " << csvFileName << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}


