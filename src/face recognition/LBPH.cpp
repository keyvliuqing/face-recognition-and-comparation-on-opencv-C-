#include "LBPH.h"
#include <fstream>
#include <sstream>
#include <iostream>

int trainAndPredictWithLBPH(const string& filename, Ptr<LBPHFaceRecognizer>& model, int& predictedLabel) {
    vector<Mat> images;
    vector<int> labels;

    // 加载和预处理图像
    ifstream file(filename.c_str(), ifstream::in);

    if (!file) {
        cerr << "Could not load file correctly..." << endl;
        return -1;
    }

    string line, path, classlabel;
    char separator = ';';

    // 读取图像并预处理
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);

        if (path.empty() || classlabel.empty()) {
            cerr << "Skipping invalid line: " << line << endl;
            continue;  // 跳过无效行
        }

        Mat img = imread(path, IMREAD_GRAYSCALE); // 以灰度模式读取图片
        if (img.empty()) {
            cerr << "Failed to load image: " << path << endl;
            continue;
        }

        // 图像预处理：直方图均衡化
        equalizeHist(img, img);

        images.push_back(img);
        labels.push_back(atoi(classlabel.c_str()));
    }

    if (images.size() < 1 || labels.size() < 1) {
        cerr << "Invalid image data..." << endl;
        return -1;
    }

    // 获取测试数据（最后一张图片）
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

    // 创建 LBPH 面部识别器对象
    model = LBPHFaceRecognizer::create();

    // 训练 LBPH 模型
    try {
        model->train(images, labels);
    }
    catch (const cv::Exception& e) {
        cerr << "Error during model training: " << e.what() << endl;
        return -1;
    }

    // 手动调整阈值
    double threshold = model->getThreshold();
    if (threshold > 100.0) {
        cerr << "Warning: Threshold is too large, resetting to 50.0." << endl;
        model->setThreshold(50.0);  // 设置一个较为合理的阈值
    }

    // 使用训练好的模型进行预测
    try {
        predictedLabel = model->predict(testSample);
    }
    catch (const cv::Exception& e) {
        cerr << "Error during prediction: " << e.what() << endl;
        return -1;
    }

    printf("Actual label: %d, Predicted label: %d\n", testLabel, predictedLabel);

    // 输出图像的长和宽
    cout << "Test image size: " << testSample.cols << " x " << testSample.rows << endl;

    // 输出 LBPH 模型参数
    printLBPHModelParams(model);

    return 0;
}

void printLBPHModelParams(Ptr<LBPHFaceRecognizer>& model) {
    // 获取并输出 LBPH 模型的参数
    int radius = model->getRadius();
    int neighbors = model->getNeighbors();
    int grid_x = model->getGridX();
    int grid_y = model->getGridY();
    double threshold = model->getThreshold();  // 更新阈值

    cout << "Model parameter summary:" << endl;
    cout << "   - Radius : " << radius << endl;
    cout << "   - Neighbors (number of neighboring pixels to consider): " << neighbors << endl;
    cout << "   - Grid size X (number of grid divisions in X direction): " << grid_x << endl;
    cout << "   - Grid size Y (number of grid divisions in Y direction): " << grid_y << endl;
    cout << "   - Threshold (sensitivity of the classifier): " << threshold << endl;
}
