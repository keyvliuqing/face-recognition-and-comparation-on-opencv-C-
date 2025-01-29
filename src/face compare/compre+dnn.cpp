/*
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
using namespace std;

// 禁用OpenCV日志输出
void disableLogOutput() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
}

// 图像预处理：调整大小、灰度化、均衡化
void preprocessImage(Mat& img) {
    resize(img, img, Size(100, 100)); // 固定大小 100x100
    if (img.channels() == 3) {
        cvtColor(img, img, COLOR_BGR2GRAY); // 转为灰度图像
    }
    equalizeHist(img, img); // 直方图均衡化
}

// 使用 DNN 提取人脸
Mat extractFaceUsingDNN(const Mat& img, const string& modelPath) {
    Net net = readNetFromCaffe(modelPath + "/deploy.prototxt", modelPath + "/res10_300x300_ssd_iter_140000.caffemodel");
    if (net.empty()) {
        cerr << "Failed to load DNN model!" << endl;
        return Mat();
    }

    Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
    net.setInput(blob);
    Mat detection = net.forward();

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.5) { // 置信度阈值
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            Rect faceRect(x1, y1, x2 - x1, y2 - y1);
            return img(faceRect).clone(); // 返回检测到的人脸区域
        }
    }
    cerr << "No face detected!" << endl;
    return Mat();
}

// 计算 PCA 特征相似度
double computePCASimilarity(const Mat& img1, const Mat& img2) {
    Mat data;
    vconcat(img1.reshape(1, 1), img2.reshape(1, 1), data); // 转换为行向量格式
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, 10); // 保留10个主成分

    Mat projected1 = pca.project(img1.reshape(1, 1));
    Mat projected2 = pca.project(img2.reshape(1, 1));

    double distance = norm(projected1, projected2, NORM_L2); // 计算欧氏距离
    return distance; // 返回欧氏距离，后续使用公式计算相似度
}

// 计算 ORB 特征点匹配的相似度
double computeORBSimilarity(const Mat& img1, const Mat& img2) {
    Ptr<ORB> orb = ORB::create(500); // 限制特征点数量500
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    if (descriptors1.empty() || descriptors2.empty()) {
        return 0.0;
    }

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    const float ratioThresh = 0.9f;
    vector<DMatch> goodMatches;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    return static_cast<double>(goodMatches.size()) / min(keypoints1.size(), keypoints2.size());
}

// 计算 LBPH 相似度
double computeLBPHSimilarity(const Mat& img1, const Mat& img2) {
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    vector<Mat> images = { img1 };
    vector<int> labels = { 0 };
    model->train(images, labels);

    int label = -1;
    double confidence = 0.0;
    model->predict(img2, label, confidence);

    return confidence; // 返回置信度，后续使用公式计算相似度
}

// 相似度转换函数
double pcaSimilarityConversion(double pcaDistance) {
    double alpha = log(6) / 5000;
    return 4.0 / (4.0 + exp(alpha * (pcaDistance - 5000)));
}

double lbphSimilarityConversion(double confidence) {
    double beta = log(6) / 50;
    return 4.0 / (4.0 + exp(beta * (confidence - 50)));
}

double orbSimilarityConversion(double orbSimilarity) {
    if (orbSimilarity < 0.7) {
        return 100 * orbSimilarity + 10;
    }
    else {
        double gamma = log(5.0 / 6.0);
        return 80 + 80 * (orbSimilarity - 0.7) * exp(gamma);
    }
}

int main() {
    // 禁用OpenCV日志输出
    disableLogOutput();

    // 测试图片路径
    string imgPath1 = "D:/opencv/xiazai.jpg";
    string imgPath2 = "D:/opencv/123.jpg";
    string modelPath = "D:/opencv/models"; 

    // 读取图片
    Mat img1 = imread(imgPath1);
    Mat img2 = imread(imgPath2);
    if (img1.empty() || img2.empty()) {
        cerr << "Failed to load images!" << endl;
        return -1;
    }

    // 使用 DNN 提取人脸
    img1 = extractFaceUsingDNN(img1, modelPath);
    img2 = extractFaceUsingDNN(img2, modelPath);

    if (img1.empty() || img2.empty()) {
        cerr << "Face extraction failed!" << endl;
        return -1;
    }

    // 图像预处理
    preprocessImage(img1);
    preprocessImage(img2);

    // 计算各个方法的原始相似度
    double pcaSimilarityRaw = computePCASimilarity(img1, img2);
    double orbSimilarityRaw = computeORBSimilarity(img1, img2);
    double lbphSimilarityRaw = computeLBPHSimilarity(img1, img2);

    // 转换相似度
    double pcaSimilarity = pcaSimilarityConversion(pcaSimilarityRaw);
    double orbSimilarity = orbSimilarityConversion(orbSimilarityRaw);
    double lbphSimilarity = lbphSimilarityConversion(lbphSimilarityRaw);

    // 第一轮判断
    int samePersonCount = 0, differentPersonCount = 0;

    if (pcaSimilarityRaw <= 6000) samePersonCount++;
    if (lbphSimilarityRaw < 50) samePersonCount++;
    if (orbSimilarityRaw > 0.5) samePersonCount++;

    if (samePersonCount >= 2) {
        cout << "Same Person" << endl;
        return 0;
    }

    if (pcaSimilarityRaw > 7500) differentPersonCount++;
    if (lbphSimilarityRaw > 80) differentPersonCount++;
    if (orbSimilarityRaw < 0.4) differentPersonCount++;

    if (differentPersonCount >= 2) {
        cout << "Different Person" << endl;
        return 0;
    }
 

    // 第二轮判断
    double finalSimilarity = 0.35 * pcaSimilarity * 100 + 0.45 * orbSimilarity + 0.3 * lbphSimilarity * 100;
  //  cout << "PCA Similarity: " << pcaSimilarity * 100 << "%" << endl;
  //  cout << "ORB Similarity: " << orbSimilarity << "%" << endl;
  //  cout << "LBPH Similarity: " << lbphSimilarity * 100 << "%" << endl;
  //  cout << "Final Similarity: " << finalSimilarity << "%" << endl;


    if (finalSimilarity > 60) {
        cout << "Same Person" << endl;
    }
    else {
        cout << "Different Person" << endl;
    }

    return 0;
}
*/