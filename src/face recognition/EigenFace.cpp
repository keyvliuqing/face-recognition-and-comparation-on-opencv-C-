#include "Eigenface.h"

EigenFaceRecognition::EigenFaceRecognition() {
    model = EigenFaceRecognizer::create();
    height = 0;
    width = 0;
}

bool EigenFaceRecognition::loadData(const string& filename) {
    ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        cerr << "Could not load file correctly..." << endl;
        return false;
    }
    string line, path, classlabel;
    char separator = ';';
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));  // 读取灰度图像
            labels.push_back(atoi(classlabel.c_str()));  // 读取标签
        }
    }
    if (images.size() < 1 || labels.size() < 1) {
        printf("invalid image path...\n");
        return false;
    }
    height = images[0].rows;
    width = images[0].cols;
    return true;
}

bool EigenFaceRecognition::trainModel() {
    try {
        model->train(images, labels);  // 训练 EigenFace 模型
    }
    catch (const cv::Exception& e) {
        cerr << "Error during model training: " << e.what() << endl;
        return false;
    }
    return true;
}

void EigenFaceRecognition::loadTestSample() {
    testSample = images.back();
    testLabel = labels.back();
    images.pop_back();
    labels.pop_back();
}

void EigenFaceRecognition::showPredictedResults() {
    // 预测
    int predictedLabel = model->predict(testSample);
    printf("Actual label: %d, Predicted label: %d\n", testLabel, predictedLabel);
}

void EigenFaceRecognition::displayResults() {
    // 获取 EigenFace 模型的相关数据
    Mat eigenvalues = model->getEigenValues();
    Mat eigenvectors = model->getEigenVectors();
    Mat mean = model->getMean();
    Mat meanFace = mean.reshape(1, height);  // 重塑为图像
    Mat dst;

    // 归一化 meanFace 到 [0, 255] 范围
    normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("Mean Face", dst);

    // 输出前 10 个 Eigenfaces
    for (int i = 0; i < min(10, eigenvectors.cols); i++) {
        Mat ev = eigenvectors.col(i).clone();
        Mat grayscale;
        Mat eigenFace = ev.reshape(1, height);  // 重塑特征向量为图像

        // 归一化 eigenFace 到 [0, 255] 范围
        normalize(eigenFace, grayscale, 0, 255, NORM_MINMAX, CV_8UC1);

        // 调整颜色映射
        Mat colorface;
        applyColorMap(grayscale, colorface, COLORMAP_BONE);  // 使用COLORMAP_BONE代替COLORMAP_JET，效果更柔和

        char winTitle[128];
        sprintf_s(winTitle, sizeof(winTitle), "eigenface_%d", i);
        imshow(winTitle, colorface);
    }

    // 进行人脸重建（从不同数量的特征脸开始）
    for (int num = min(10, eigenvectors.cols); num < min(eigenvectors.cols, 300); num += 30) {
        Mat evs = Mat(eigenvectors, Range::all(), Range(0, num));  // 选取前 num 个特征脸
        Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));  // 投影
        Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);  // 重建
        Mat result = reconstruction.reshape(1, height);  // 重塑为图像

        // 归一化重建图像
        normalize(result, reconstruction, 0, 255, NORM_MINMAX, CV_8UC1);

        char winTitle[128];
        sprintf_s(winTitle, sizeof(winTitle), "reconstructed_face_%d", num);
        imshow(winTitle, reconstruction);
    }

    waitKey(0);
}

void EigenFaceRecognition::predictAndDisplayResults() {
    loadTestSample();
    showPredictedResults();
    displayResults();
}
