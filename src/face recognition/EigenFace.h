#pragma once
#ifndef EIGENFACE_H
#define EIGENFACE_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace cv::face;
using namespace std;

class EigenFaceRecognition {
public:
    EigenFaceRecognition();
    bool loadData(const string& filename);
    bool trainModel();
    void predictAndDisplayResults();
private:
    vector<Mat> images;
    vector<int> labels;
    Ptr<EigenFaceRecognizer> model;
    int height;
    int width;
    Mat testSample;
    int testLabel;
    void loadTestSample();
    void showPredictedResults();
    void displayResults();
};

#endif // EIGENFACE_H
