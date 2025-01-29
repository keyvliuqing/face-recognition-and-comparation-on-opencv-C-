#include "HaarFaceDetector.h"
#include "createDirectory.h" // 引入封装的目录创建函数
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Preprocessing.h" 

using namespace std;
using namespace cv;

void detectAndSaveFaces(const std::string& haarCascadePath, const std::string& savePath)
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

    CascadeClassifier faceDetector;
    if (!faceDetector.load(haarCascadePath))
    {
        cerr << "Failed to load Haar cascade file." << endl;
        return;
    }

    namedWindow("Haar Face Detection", WINDOW_AUTOSIZE);
    Mat frame;
    vector<Rect> faces;
    int count = 0;

   
    Size targetSize(300, 300); 

    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cerr << "Empty frame. Exiting..." << endl;
            break;
        }

        flip(frame, frame, 1); // 镜像翻转

        // 调用预处理函数
       //  Mat preprocessedFrame = Preprocessing::preprocessFace(frame, targetSize); 

        faceDetector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(30, 30));

        for (size_t i = 0; i < faces.size(); i++)
        {
            rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
            Mat faceROI = frame(faces[i]);

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

            // 保存图片并检查结果
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

        imshow("Haar Face Detection", frame);
        char c = waitKey(10);
        if (c == 27) // 按 ESC 键退出
        {
            break;
        }
    }

    capture.release();
    destroyAllWindows();
}
