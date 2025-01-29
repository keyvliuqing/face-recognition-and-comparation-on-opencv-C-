#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/face.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>  // 添加filesystem头文件

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
using namespace std;
namespace fs = std::filesystem;  // 定义命名空间别名

// 添加全局路径变量
string modelDir;

// Window Procedure Function (Windows GUI Event Handler)
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

// Global variables for UI components
HWND hLeftImageLabel, hRightImageLabel, hCompareButton, hLeftButton, hRightButton;
Mat leftImage, rightImage;
wstring leftImagePath, rightImagePath;


void disableLogOutput() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
}

void preprocessImage(Mat& img) {
    resize(img, img, Size(300, 300));  // Resize to a fixed size
    if (img.channels() == 3) {
        cvtColor(img, img, COLOR_BGR2GRAY);  // Convert to grayscale
    }
    equalizeHist(img, img);  // Apply histogram equalization
}

Mat extractFaceUsingDNN(const Mat& img, const string& modelPath) {
    // Load the DNN model
    Net net = readNetFromCaffe(modelPath + "/deploy.prototxt", modelPath + "/res10_300x300_ssd_iter_140000.caffemodel");
    if (net.empty()) {
        cerr << "Failed to load DNN model!" << endl;
        return Mat();
    }

    // Prepare the image for DNN processing
    Mat blob = blobFromImage(img, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
    net.setInput(blob);
    Mat detection = net.forward();

    // Detect faces in the image
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.5) {  // If the confidence is above threshold
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);
            Rect faceRect(x1, y1, x2 - x1, y2 - y1);
            return img(faceRect).clone();  // Return the face region
        }
    }
    cerr << "No face detected!" << endl;
    return Mat();
}

double computePCASimilarity(const Mat& img1, const Mat& img2) {
    // Compute PCA similarity between two images
    Mat data;
    vconcat(img1.reshape(1, 1), img2.reshape(1, 1), data);
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, 10);
    Mat projected1 = pca.project(img1.reshape(1, 1));
    Mat projected2 = pca.project(img2.reshape(1, 1));
    return norm(projected1, projected2, NORM_L2);  // Compute L2 norm between projected images
}

double computeORBSimilarity(const Mat& img1, const Mat& img2) {
    Ptr<ORB> orb = ORB::create(500);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

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

double computeLBPHSimilarity(const Mat& img1, const Mat& img2) {
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    vector<Mat> images = { img1 };
    vector<int> labels = { 0 };
    model->train(images, labels);
    int label = -1;
    double confidence = 0.0;
    model->predict(img2, label, confidence);
    return confidence;
}


void OnStartComparison(HWND hwnd) {
    // Load images
    leftImage = imread(string(leftImagePath.begin(), leftImagePath.end()));
    rightImage = imread(string(rightImagePath.begin(), rightImagePath.end()));
    if (leftImage.empty() || rightImage.empty()) {
        MessageBox(hwnd, L"Failed to load images!", L"Error", MB_OK);
        return;
    }

    // 修改这里使用全局路径变量
    leftImage = extractFaceUsingDNN(leftImage, modelDir);
    rightImage = extractFaceUsingDNN(rightImage, modelDir);

    if (leftImage.empty() || rightImage.empty()) {
        MessageBox(hwnd, L"Face extraction failed!", L"Error", MB_OK);
        return;
    }

    // Preprocess images
    preprocessImage(leftImage);
    preprocessImage(rightImage);

    // Calculate similarities using PCA, ORB, and LBPH
    double pcaSimilarityRaw = computePCASimilarity(leftImage, rightImage);
    double orbSimilarityRaw = computeORBSimilarity(leftImage, rightImage);
    double lbphSimilarityRaw = computeLBPHSimilarity(leftImage, rightImage);

    // First round of comparisons
    int samePersonCount = 0, differentPersonCount = 0;

    if (pcaSimilarityRaw <= 6000) samePersonCount++;
    if (lbphSimilarityRaw < 50) samePersonCount++;
    if (orbSimilarityRaw > 0.5) samePersonCount++;

    if (samePersonCount >= 2) {
        MessageBox(hwnd, L"Same Person", L"Result", MB_OK);
        return;
    }

    if (pcaSimilarityRaw > 7500) differentPersonCount++;
    if (lbphSimilarityRaw > 80) differentPersonCount++;
    if (orbSimilarityRaw < 0.4) differentPersonCount++;

    if (differentPersonCount >= 2) {
        MessageBox(hwnd, L"Different Person", L"Result", MB_OK);
        return;
    }

    // Second round of comparisons with weighted final similarity
    double pcaSimilarity = computePCASimilarity(leftImage, rightImage);
    double orbSimilarity = computeORBSimilarity(leftImage, rightImage);
    double lbphSimilarity = computeLBPHSimilarity(leftImage, rightImage);

    // Weighted similarity calculation
    double finalSimilarity = 0.35 * pcaSimilarity * 100 + 0.45 * orbSimilarity + 0.3 * lbphSimilarity * 100;

    if (finalSimilarity > 60) {
        MessageBox(hwnd, L"Same Person", L"Result", MB_OK);
    }
    else {
        MessageBox(hwnd, L"Different Person", L"Result", MB_OK);
    }
}

// File dialog for image selection
void LoadImage(HWND hwnd, bool isLeftImage) {
    OPENFILENAME ofn;
    wchar_t szFile[260];
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = hwnd;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = L'\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = L"Image Files\0*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.pgm\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if (GetOpenFileName(&ofn) == TRUE) {
        wstring filePath = ofn.lpstrFile;
        if (isLeftImage) {
            leftImagePath = filePath;
            SetWindowTextW(hLeftImageLabel, leftImagePath.c_str());
        }
        else {
            rightImagePath = filePath;
            SetWindowTextW(hRightImageLabel, rightImagePath.c_str());
        }
    }
}

// Create Window

int main() {
    disableLogOutput();

    // 设置当前执行路径
    modelDir = fs::current_path().string();

    const wchar_t CLASS_NAME[] = L"Face Compare Window";
    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowExW(0, CLASS_NAME, L"Face Compare", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600, NULL, NULL, wc.hInstance, NULL);

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    // Message loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

// Window Procedure for handling UI events
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_COMMAND:
        if ((HWND)lParam == hLeftButton) {
            LoadImage(hwnd, true);
        }
        else if ((HWND)lParam == hRightButton) {
            LoadImage(hwnd, false);
        }
        else if ((HWND)lParam == hCompareButton) {
            OnStartComparison(hwnd);
        }
        break;

    case WM_CREATE:
        hLeftImageLabel = CreateWindowW(L"Static", L"Left Image", WS_CHILD | WS_VISIBLE, 10, 10, 350, 50, hwnd, NULL, NULL, NULL);
        hRightImageLabel = CreateWindowW(L"Static", L"Right Image", WS_CHILD | WS_VISIBLE, 10, 70, 350, 50, hwnd, NULL, NULL, NULL);

        hLeftButton = CreateWindowW(L"Button", L"Load Left Image", WS_CHILD | WS_VISIBLE, 370, 10, 120, 30, hwnd, NULL, NULL, NULL);
        hRightButton = CreateWindowW(L"Button", L"Load Right Image", WS_CHILD | WS_VISIBLE, 370, 70, 120, 30, hwnd, NULL, NULL, NULL);

        hCompareButton = CreateWindowW(L"Button", L"Compare", WS_CHILD | WS_VISIBLE, 370, 130, 120, 30, hwnd, NULL, NULL, NULL);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    return 0;
}

