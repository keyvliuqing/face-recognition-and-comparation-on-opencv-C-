#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace dnn;
using namespace std;
namespace fs = std::filesystem;

int main()
{
    // 获取当前exe文件所在的路径
    fs::path exeDir = fs::current_path();

    // 构建相对路径
    std::string modelPath = (exeDir / "res10_300x300_ssd_iter_140000.caffemodel").string(); // 模型文件路径
    std::string configPath = (exeDir / "deploy.prototxt").string(); // 配置文件路径

    // 打开摄像头
    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << "Could not open camera..." << endl;
        return -1;
    }

    // 加载DNN模型
    Net net = readNetFromCaffe(configPath, modelPath);
    if (net.empty())
    {
        cerr << "Failed to load DNN model." << endl;
        return -1;
    }

    // 设置目标图像大小
    Size targetSize(300, 300);  // 可以根据需要修改目标大小
    namedWindow("DNN Face Detection", WINDOW_AUTOSIZE);
    Mat frame;

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cerr << "Empty frame. Exiting..." << endl;
            break;
        }

        flip(frame, frame, 1); // 镜像翻转

        // 构建DNN输入
        Mat blob = blobFromImage(frame, 1.0, targetSize, Scalar(104, 117, 123), false, false);
        net.setInput(blob);

        // 获取人脸检测结果
        Mat detections = net.forward();

        // 解析检测结果
        Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.5)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect faceBox(Point(x1, y1), Point(x2, y2));
                rectangle(frame, faceBox, Scalar(0, 255, 0), 2); // 画出人脸框
            }
        }

        // 显示处理后的图像
        imshow("DNN Face Detection", frame);

        char c = waitKey(10);
        if (c == 27) // 按 ESC 键退出
        {
            break;
        }
    }

    // 释放摄像头并销毁窗口
    capture.release();
    destroyAllWindows();

    return 0;
}
