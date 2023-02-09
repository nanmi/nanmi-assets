#ifndef __VIBE__
#define __VIBE__


#include<iostream>
#include<cstdio>
#include<opencv2/opencv.hpp>
 
 
using namespace cv;
using namespace std;
 
 
//每个像素点的样本个数默认值
#define DEFAULT_NUM_SAMPLES 20

//#min指数默认值
#define DEFAULT_MIN_MATCHES 2

//Sqthere半径默认值
#define DEFAULT_RADIUS 20

//子采样概率默认值
#define DEFAULT_RANDOM_SAMPLE 16
 

class ViBe
{
public:
    ViBe(int num_sam = DEFAULT_NUM_SAMPLES,
        int min_match = DEFAULT_MIN_MATCHES,
        int r = DEFAULT_RADIUS,
        int rand_sam = DEFAULT_RANDOM_SAMPLE);
    ~ViBe(void);

    //背景模型初始化
    void init(Mat img);

    //处理第一帧图像
    void ProcessFirstFrame(Mat img);

    //运行ViBe算法，提取前景区域并更新背景模型样本库
    void Run(Mat img);

    //获取前景模型二值图像
    Mat getFGModel();

    //删除样本库
    void deleteSamples();

    //x的邻居点
    int c_xoff[9];

    //y的邻居点
    int c_yoff[9];

private:
    //样本库
    unsigned char*** samples;

    //前景模型二值图像
    Mat FGModel;

    //每个像素点的样本个数
    int num_samples;

    //#min指数
    int num_min_matches;

    //Sqthere半径
    int radius;

    //子采样概率
    int random_sample;
};

#endif