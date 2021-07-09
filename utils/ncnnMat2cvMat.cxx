/*
*  - ncnn中的数据是float类型.

*  - cvMat的类型是CV_8UC3, mat.data指定的类型是char *型, 故cvMat可以用下标[]直接索引.

*  - ncnn中数据的排列格式为(channel, h, w), cv::Mat中数据的排列格式为(h, w, channel).

*  - cv::Mat中颜色顺序为BGR, ncnn::Mat格式为BGR.

*/

#include <opencv2/opencv.hpp>
#include <net.h>
#include <benchmark.h>


void ncnnMat2cvMat(ncnn::Mat& ncnnMat, cv::Mat& cvMat)
{
    cv::Mat cvMat(ncnnMat.h, ncnnMat.w, CV_8UC3);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < ncnnMat.h; i++)
        {
            for (int j = 0; j < ncnnMat.w; j++)
            {
                float t = ((float*)ncnnMat.data)[j + i * ncnnMat.w + c * ncnnMat.h * ncnnMat.w];
                cvMat.data[(2 - c) + j * 3 + i * ncnnMat.w * 3] = t;
            }
        }
    }
    cvMat = cvMat;
}
