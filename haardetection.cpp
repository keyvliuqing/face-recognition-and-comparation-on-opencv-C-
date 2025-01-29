#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int main()
{
    // 获取当前exe文件所在的路径
    fs::path exeDir = fs::current_path();

    // 构建相对路径（假设haarcascade文件与exe文件在同一目录下）
    std::string haarCascadePath = (exeDir / "haarcascade_frontalface_default.xml").string(); // Haar模型文件路径

    // 打开摄像头
    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << "Could not open camera..." << endl;
        return -1;
    }

    // 加载Haar级联分类器
    CascadeClassifier faceDetector;
    if (!faceDetector.load(haarCascadePath))
    {
        cerr << "Failed to load Haar cascade file." << endl;
        return -1;
    }

    // 设置目标图像大小
    Size targetSize(300, 300);  // 可以根据需要修改目标大小
    namedWindow("Haar Face Detection", WINDOW_AUTOSIZE);
    Mat frame;
    vector<Rect> faces;

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cerr << "Empty frame. Exiting..." << endl;
            break;
        }

        flip(frame, frame, 1); // 镜像翻转

        // 人脸检测
        faceDetector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(30, 30));

        // 绘制检测到的人脸框
        for (size_t i = 0; i < faces.size(); i++)
        {
            rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
        }

        // 显示处理后的图像
        imshow("Haar Face Detection", frame);

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
