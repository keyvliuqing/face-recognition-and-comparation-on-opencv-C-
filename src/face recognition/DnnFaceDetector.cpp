#include "DnnFaceDetector.h" 
#include "createDirectory.h"
#include "Preprocessing.h" 
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

void detectAndSaveFacesDNN(const std::string& modelPath, const std::string& configPath, const std::string& savePath)
{
    // 创建保存目录
    if (!createDirectory(savePath))
    {
        cerr << "Failed to create directory: " << savePath << endl;
        return;
    }

    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << "Could not open camera..." << endl;
        return;
    }

    Net net = readNetFromCaffe(configPath, modelPath);
    if (net.empty())
    {
        cerr << "Failed to load DNN model." << endl;
        return;
    }

    namedWindow("DNN Face Detection", WINDOW_AUTOSIZE);
    Mat frame;
    int count = 0;

    // 设置统一的目标尺寸,选择 300x300 作为目标尺寸
    Size targetSize(300, 300); // 你可以根据需要修改目标大小

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cerr << "Empty frame. Exiting..." << endl;
            break;
        }

        flip(frame, frame, 1); // 镜像翻转
       // 调用预处理函数
      //   Mat preprocessedFrame = Preprocessing::preprocessFace(frame, targetSize); 

        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 117, 123), false, false);
        net.setInput(blob);
        Mat detections = net.forward();

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
                rectangle(frame, faceBox, Scalar(0, 255, 0), 2);

                Mat faceROI = frame(faceBox);

                // 检查 ROI 是否有效
                if (faceROI.empty() || faceROI.cols <= 0 || faceROI.rows <= 0)
                {
                    cerr << "Invalid face ROI. Skipping..." << endl;
                    continue;
                }

                // 将人脸图像调整为目标尺寸
                Mat resizedFace;
                resize(faceROI, resizedFace, targetSize);

                // 文件名生成（添加 .jpg 扩展名）
                string filename = savePath + "/face_" + to_string(count) + ".jpg";
                cout << "Attempting to save: " << filename << endl;

                try
                {
                    if (!imwrite(filename, resizedFace))
                    {
                        cerr << "Failed to save image: " << filename << endl;
                    }
                    else
                    {
                        cout << "Saved: " << filename << endl;
                        count++;
                    }
                }
                catch (const cv::Exception& e)
                {
                    cerr << "Exception during imwrite: " << e.what() << endl;
                }
            }
        }

        imshow("DNN Face Detection", frame);

        char c = waitKey(10);
        if (c == 27) // 按 ESC 键退出
        {
            break;
        }
    }

    capture.release();
    destroyAllWindows();
}
