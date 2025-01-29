#include <iostream>

//csv文件生成
/*
#include "CsvGenerator.h"
int main()
{
    std::string directory = "D:/opencv/ATT Face/s1"; //图片路径
    std::string csvFileName = "D:/opencv/ATT Face/s1/image.csv";//要生成csv文件的路径

    if (generateCsv(directory, csvFileName))
    {
        std::cout << "CSV generated successfully!" << std::endl;
    }
    else
    {
        std::cerr << "Failed to generate CSV!" << std::endl;
    }

    return 0;
}
*/


// dnn人脸检测调用
/*
#include "DnnFaceDetector.h"
#include <string>
int main()
{
    std::string modelPath = "D:/opencv/models/res10_300x300_ssd_iter_140000.caffemodel";//res10_300x300_ssd_iter_140000.caffemodel模型路径
    std::string configPath = "D:/opencv/models/deploy.prototxt";//deploy.prototx模型路径
    std::string savePath = "D:/opencv/out_face/Preprocessing";//保存图片的路径

    detectAndSaveFacesDNN(modelPath, configPath, savePath);

    return 0;
}
*/



// haar人脸识别调用
/*
#include "HaarFaceDetector.h"
int main()
{
    std::string haarCascadePath = "D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml";//haarcascade_frontalface_default.xml模型路径
    std::string savePath = "D:/opencv/out_face/Preprocessing";//保存图片的路径

    detectAndSaveFaces(haarCascadePath, savePath);

    return 0;
}
*/


//LBPH调用
/*
#include "LBPH.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

int main() {
   // 禁用 OpenCV 的日志输出
   cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    string filename = "D:/opencv/out_face/dnn/image.csv";//保存csv文件的路径
    Ptr<LBPHFaceRecognizer> model;
    int predictedLabel;

    // 训练和预测
    if (trainAndPredictWithLBPH(filename, model, predictedLabel) != 0) {
        return -1;
    }

    // 你也可以在这里添加其他操作，比如显示预测结果
    waitKey(0); // 等待键盘输入
    return 0;
}
*/

// EigenFace 调用
/*
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include "Eigenface.h"

using namespace cv;
using namespace std;

int main() {
    // 禁用 OpenCV 的日志输出
     cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // 创建 EigenFaceRecognition 对象
    EigenFaceRecognition eigenfaceRecognizer;

    // 加载数据并训练模型
    string filename = "D:/opencv/out_face/dnn/image.csv";//保存csv文件的路径
    if (!eigenfaceRecognizer.loadData(filename) || !eigenfaceRecognizer.trainModel()) {
        return -1;
    }

    // 预测并显示结果
    eigenfaceRecognizer.predictAndDisplayResults();

    return 0;
}
*/


// FisherFace 调用
/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include "FisherFace.h"

using namespace cv;
using namespace std;

int main() {
   // 禁用 OpenCV 的日志输出
   cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // 创建 FisherFaceRecognition 对象
    FisherFaceRecognition fisherFaceRecognizer;

    // 加载数据并训练模型
    string filename = "D:/opencv/out_face/dnn/image.csv";//保存csv文件的路径
    if (!fisherFaceRecognizer.loadData(filename) || !fisherFaceRecognizer.trainModel()) {
        return -1;
    }

    // 预测并显示结果
    fisherFaceRecognizer.predictAndDisplayResults();

    return 0;
}
*/

