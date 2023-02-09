#include "vibe.h"
/*
构造函数ViBe
参数：
int num_sam:每个像素点的样本个数
int min_match:#min 指数
int r: Sqthere 半径
int rand_sam:子采样概率
*/

ViBe::ViBe(int num_sam, int min_match, int r, int rand_sam)
{
    num_samples = num_sam;
    num_min_matches = min_match;
    radius = r;
    random_sample = rand_sam;
    int c_off[9] = { -1,0,1,-1,1,-1,0,1,0 };
    for (int i = 0; i < 9; i++) {
        c_xoff[i] = c_yoff[i] = c_off[i];
    }
}

/*析构函数：~ViBe
  说明：释放样本库内存
*/
ViBe::~ViBe(void)
{
    deleteSamples();
}
 
 
/*
  函数名init
  说明：背景模型初始化
       为样本库分配空间
  参数：Mat img:源图像
  返回值：void
*/
void ViBe::init(Mat img)
{
    //动态分配三维数组，samples[][][num_samples]存储前景被连续检测的次数
    //
    samples = new unsigned char** [img.rows];
    for (int i = 0; i < img.rows; i++)
    {
        samples[i] = new uchar * [img.cols];
        for (int j = 0; j < img.cols; j++)
        {
            //数组中，在num_samples之外多增加的一个值，用于统计该像素点连续成为前景的次数
            samples[i][j] = new uchar[num_samples + 1];
            for (int k = 0; k < num_samples + 1; k++)
            {
                //创建样本库是，所有样本全部初始化为0
                samples[i][j][k] = 0;
            }
        }
    }

    FGModel = Mat::zeros(img.size(), CV_8UC1);
}


/*
     函数名 ProcessFirstFrame
     说明：处理第一帧图像
         读取视频序列第一帧，并随机选取像素点邻域内像素填充样本库，初始化背景模型
    参数：
    Mat img:源图像
    返回值：void
     */
void ViBe::ProcessFirstFrame(Mat img)
{
    RNG rng;
    int row, col;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int k = 0; k < num_samples; k++)
            {
                //随机选择num_samples个邻域像素点，构建背景模型
                int random;
                random = rng.uniform(0, 9); row = i + c_yoff[random];
                random = rng.uniform(0, 9); col = j + c_xoff[random];

                //防止选取的像素点越界
                if (row < 0)
                    row = 0;
                if (row >= img.rows)
                    row = img.rows - 1;
                if (col < 0)
                    col = 0;
                if (col >= img.cols)
                    col = img.cols - 1;
 
                //为样本库赋值随机值
                samples[i][j][k] = img.at<uchar>(row, col);
            }
        }
    }
}
 
 
/*
函数名：Run
说明：运行ViBe算法，提取前景区域并更新背景模型样本库
参数：
Mat img 源图像
返回值：void
*/
void ViBe::Run(Mat img)
{
    RNG rng;
    int k = 0, dist = 0, matches = 0;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            //前景提取
            //说明：计算当前像素值与样本库的匹配情况
            //参数：
            //int matches:当前像素值与   样本库中值之差小于阈值范围RADIUS的个数
            //int count: 遍历样本库的缓存变量
            for (k = 0, matches = 0; matches < num_min_matches && k < num_samples; k++)
            {
                dist = abs(samples[i][j][k] - img.at<uchar>(i, j));
                if (dist < radius)
                    matches++;
            }
            //说明：当前像素值与样本库中值匹配次数较高，则认为是背景像素点；
            //此时更新前景统计次数、更新前景模型、更新该像素模型样本值、更新该像素点邻域像素点
            if (matches >= num_min_matches)
            {
                //已经认为是背景像素，故该像素前景统计次数置0
                samples[i][j][num_samples] = 0;

                //该像素点的前景模型像素值置0
                FGModel.at<uchar>(i, j) = 0;
            }
            //说明：当前像素值与样本库中值匹配次数较低，则认为是前景像素点
            //此时需要更新前景统计次数，判断更新前景模型
            else {
                //已经认为是前景像素，故该像素的前景统计次数+1
                samples[i][j][num_samples]++;

                //该像素点的前景模型像素值置255
                FGModel.at<uchar>(i, j) = 255;

                //如果某个像素点连续50次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
                if (samples[i][j][num_samples] > 50)
                {
                    int random = rng.uniform(0, num_samples);
                    samples[i][j][random] = img.at<uchar>(i, j);
                }
            }
            //更新模型样本库
            if (matches >= num_min_matches)
            {
                //已经认为该像素是背景像素，那么它有1/φ的概率去更新自己的模型样本值
                int random = rng.uniform(0, random_sample);
                if (random == 0)
                {
                    random = rng.uniform(0, num_samples);
                    samples[i][j][random] = img.at<uchar>(i, j);
                }
                //同时也有1/φ的概率去更新它的邻居点的模型样本值
                random = rng.uniform(0, random_sample);
                if (random == 0)
                {
                    int row, col;
                    random = rng.uniform(0, 9); row = i + c_yoff[random];
                    random = rng.uniform(0, 9); col = j + c_xoff[random];

                    //防止选取的像素点越界
                    if (row < 0)
                        row = 0;
                    if (row >= img.rows)
                        row = img.rows - 1;
                    if (col < 0)
                        col = 0;
                    if (col >= img.cols)
                        col = img.cols - 1;

                    //为样本库赋值随机值

                    random = rng.uniform(0, num_samples);
                    samples[row][col][random] = img.at<uchar>(i, j);
                }
            }
        }
    }
}

/*
函数名 ：getFGModel
说明：获取前景模型二值图像
返回值：Mat
*/
Mat ViBe::getFGModel()
{
    return FGModel;
}
 
 
/*
函数名：deletesamples
说明：删除样本库
返回值：void
*/
void ViBe::deleteSamples()
{
    delete samples;
}
