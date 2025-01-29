/*
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <iomanip> // 用于格式化百分比输出

void preprocessImage(cv::Mat& img) {
    // 调整大小到统一规格
    int targetWidth = 500;
    double scale = static_cast<double>(targetWidth) / img.cols;
    cv::resize(img, img, cv::Size(targetWidth, static_cast<int>(img.rows * scale)));

    // 光照调整：直方图均衡化
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    cv::equalizeHist(img, img);
}

void displayMatchingResult(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& goodMatches,
    double similarity) {
    // 绘制特征点匹配
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches);

    // 显示特征点匹配结果
    cv::imshow("Feature Matching", imgMatches);
    std::cout << std::fixed << std::setprecision(2); // 设置输出两位小数
    std::cout << "Similarity Score: " << similarity * 100 << "%" << std::endl;

    // 输出判断结果
    double threshold = 0.5; // 宽松阈值
    if (similarity > threshold) {
        std::cout << "Judgement: Same Person" << std::endl;
    }
    else {
        std::cout << "Judgement: Different Person" << std::endl;
    }
}

int main() {
    // 测试图片路径
    std::string imgPath1 = "D:/opencv/out_face/dnn/face_0.jpg";
    std::string imgPath2 = "D:/opencv/ATT Face/s23/6.pgm";

    // 读取图片
    cv::Mat img1 = cv::imread(imgPath1);
    cv::Mat img2 = cv::imread(imgPath2);
    if (img1.empty() || img2.empty()) {
        std::cerr << "Failed to load images!" << std::endl;
        return -1;
    }

    // 对图片进行预处理
    preprocessImage(img1);
    preprocessImage(img2);

    // 提取特征点和描述符 (ORB特征)
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1500); // 提取更多关键点
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // 特征点匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Lowe's Ratio Test
    const float ratioThresh = 0.9f; // 放宽过滤阈值
    std::vector<cv::DMatch> goodMatches;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    // 计算相似度（基于有效匹配点数量）
    double similarity = 0.0;
    if (!keypoints1.empty() && !keypoints2.empty()) {
        similarity = static_cast<double>(goodMatches.size()) / std::min(keypoints1.size(), keypoints2.size());
    }

    // 显示匹配结果并判断
    displayMatchingResult(img1, img2, keypoints1, keypoints2, goodMatches, similarity);

    // 等待键盘输入关闭窗口
    cv::waitKey(0);
    return 0;
}
*/