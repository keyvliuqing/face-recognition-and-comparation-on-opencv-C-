#pragma once
#ifndef FISHERFACE_H
#define FISHERFACE_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;

class FisherFaceRecognition {
public:
    FisherFaceRecognition();
    ~FisherFaceRecognition();

    bool loadData(const string& filename);
    bool trainModel();
    void predictAndDisplayResults();

private:
    Ptr<FisherFaceRecognizer> model;
    vector<Mat> images;
    vector<int> labels;
    int height, width;
};

#endif // FISHERFACE_H
