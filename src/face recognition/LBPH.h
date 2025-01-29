#pragma once
#ifndef LBPH_H
#define LBPH_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;


int trainAndPredictWithLBPH(const string& filename, Ptr<LBPHFaceRecognizer>& model, int& predictedLabel);
void printLBPHModelParams(Ptr<LBPHFaceRecognizer>& model);

#endif // LBPH_H

