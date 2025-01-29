#pragma once
#ifndef DNN_FACE_DETECTOR_H
#define DNN_FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

void detectAndSaveFacesDNN(const std::string& modelPath, const std::string& configPath, const std::string& savePath);

#endif // DNN_FACE_DETECTOR_H
