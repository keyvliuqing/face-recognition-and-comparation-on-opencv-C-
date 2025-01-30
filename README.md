本项目基于 OpenCV C++，主要介绍人脸识别相关应用的开发过程，包括人脸检测和人脸对比，以及不同算法的实现方式。

📂 项目结构
1️⃣ face_recognition/ (主分支)
包含三个可执行程序：
haardetection.exe：使用 OpenCV 自带的 Haar 级联分类器进行实时人脸检测。
dnndetection.exe：使用 DNN（深度神经网络）进行实时人脸检测。

Haar 与 DNN 的区别：
Haar：基于特征模板检测，速度较快，但对光照、角度等因素敏感，容易误检。
DNN：基于深度学习模型，检测精度较高，适应性更强，但计算开销较大。
运行要求：请确保摄像头权限已开启。

FaceCompare.exe：用于导入两张人脸图片并判断是否为同一人（注意：图片路径需为全英文）。
算法介绍：
结合 ORB、PCA、LBPH 三种人脸识别方法，进行特征值提取与匹配。
识别思路：
第一轮筛选：基于三种特征计算相似度，排除明显相同或不同的情况。
第二轮判断：针对无法直接判定的情况，利用幂指数函数建模，调整参数权重，使不同特征的相似度值趋于统一。
最终决策：采用加权平均计算综合相似度，设定 60% 以上判定为同一人。

2️⃣ src/ (源码文件)
包含项目的完整源码，供学习交流：
dnndetection.cpp & haardetection.cpp：人脸检测相关代码。
face_compare/：人脸对比的核心代码，从 ORB 算法 到 结合 DNN，再到最终实现（详见 main.cpp）。
face_recognition/：人脸识别相关代码，包括：CSV 文件创建、Haar & DNN 检测、特征提取（EigenFace、FisherFace、LBPH）
采用 Visual Studio 三段封装（头文件 + 源文件 + main 函数）

💻 运行环境
1️⃣ 开发环境：
Visual Studio 2022
[OpenCV 4.1.0（官网下载）](https://opencv.org/releases/)

opencv 拓展模块：[opencv_contrib](https://github.com/opencv/opencv_contrib)

2️⃣ OpenCV 配置（CMake 构建）
安装 CMake 并构建 OpenCV：
打开 CMake
Source 选择 OpenCV 源码路径
Build 选择 目标构建路径
Configure → 选择 Visual Studio x64 → Finish
勾选 BUILD_opencv_world（简化 VS 的库链接）
Generate → Open Project

在 Visual Studio 中：
运行 ALL_BUILD
然后运行 INSTALL

VS 项目配置：
包含目录：install/include
库目录：install/x64/v15/lib（或 v16/v17）
附加依赖项：opencv_world410.lib（以及其他lib）
用头文件测试是否配置成功：
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

3️⃣ DNN 人脸检测模型
本项目使用的当前使用的两个DNN模型：
[DNN 人脸检测模型](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)

[下载 Caffe 模型](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

⚡ 适用范围与局限性
✅ 适用场景：
适用于 标准人脸数据库 或 固定条件下的人脸比对。
适用于 Haar/DNN 检测到的人脸图像进行对比。
❌ 局限性：
由于采用 传统特征匹配 而非深度学习，对 不同光照、角度、表情变化适应性较差，即使进行预处理也难以消除所有影响。
🚀 未来开发方向
为了提升识别精度和适应性，未来计划：
集成深度学习模型：
CNN（卷积神经网络）
预训练人脸识别模型（FaceNet、VGGFace、ArcFace）
优化特征提取方法，提升对不同拍摄条件的鲁棒性。
📌 致谢
感谢 OpenCV 及相关开源贡献者提供的DNN模型与扩展库！
