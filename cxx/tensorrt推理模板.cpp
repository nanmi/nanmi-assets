#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include "logging.h"
#include "postprocess_op.h"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

using namespace nvinfer1;

static Logger gLogger;

cv::Mat static_resize(cv::Mat& img, bool keep, int target_w, int target_h, float& scale) {
    cv::Mat cropped;
    if (keep)
    {
        scale = cv::min(float(target_w)/img.cols, float(target_h)/img.rows);
        auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

        cv::Mat resized;
        cv::resize(img, resized, scaleSize,0,0);

        cropped = cv::Mat::zeros(target_h, target_w, CV_8UC3);
        cv::Rect rect((target_w - scaleSize.width)/2, (target_h-scaleSize.height)/2, scaleSize.width, scaleSize.height);
        resized.copyTo(cropped(rect));
    }
    else
    {
        auto scaleSize = cv::Size(target_w, target_h);
        cv::resize(img, cropped, scaleSize,0,0);
    }
    
    return cropped;
}

void ResizeImgType0(const cv::Mat &img, cv::Mat &resize_img,
                         int max_size_len, float &ratio_h, float &ratio_w) {
  int w = img.cols;
  int h = img.rows;

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }

  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);

  resize_h = max(int(round(float(resize_h) / 32) * 32), 32);
  resize_w = max(int(round(float(resize_w) / 32) * 32), 32);

  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  ratio_h = float(resize_h) / float(h);
  ratio_w = float(resize_w) / float(w);
}

std::vector<std::vector<std::vector<int>>>
DBPostProcess(float* out_data, cv::Mat &srcimg, float &ratio_h, float &ratio_w, int n2, int n3)
{
    // int n2 = 640;
    // int n3 = 640;
    int n = n2 * n3;
    // post-process
    PostProcessor post_processor_;

    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
        pred[i] = float(out_data[i]);
        cbuf[i] = (unsigned char)((out_data[i]) * 255);
    }

    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());
    cv::imwrite("cbuf.jpg", cbuf_map);
    cv::imwrite("pref.jpg", pred_map);
    const double threshold = 0.3 * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (false) {
        cv::Mat dila_ele =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    std::vector<std::vector<std::vector<int>>> boxes = post_processor_.BoxesFromBitmap(pred_map, bit_map, 0.5f, 2.0f, "show");

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
    return boxes;
}



const float mean_vals[3] = {0.5f, 0.5f, 0.5f};
const float norm_vals[3] = {0.5f, 0.5f, 0.5f};
float* blobFromImage(cv::Mat& img)
{
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    for (int c = 0; c < channels; c++) 
    {
        for (int  h = 0; h < img_h; h++) 
        {
            for (int w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f - mean_vals[c]) / norm_vals[c];
            }
        }
    }
    return blob;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./ocr_det ../ch-pp-ocrv3-det-fp16.engine -i ../dog.jpg  // run inference" << std::endl;
        return -1;
    }

    const std::string input_image_path {argv[3]};
    // const std::string img_dir {argv[3]};

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // std::vector<std::string> file_names;
    // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //     std::cerr << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }

    // GPU pointer
    int nbBindings = 2;
    float* buffers[nbBindings];
    const int batch_size = 1;
    const int i_idx = engine->getBindingIndex("x"); //1x3x550x550
    const int loc_idx= engine->getBindingIndex("sigmoid_0.tmp_0"); // 1x19248x4
    std::chrono::steady_clock::time_point Tbegin, Iend, Tend;


    cv::Mat src_bgr = cv::imread(input_image_path, 1); // bgr
    cv::Mat src_rgb;
    cv::cvtColor(src_bgr, src_rgb, cv::COLOR_BGR2RGB);  // to rgb

    // // stuff we know about the network and the input blobs size
    // static const int INPUT_W = 640;
    // static const int INPUT_H = 640;
    // float scale{};
    // cv::Mat pr_img = static_resize(src_bgr, false, INPUT_W, INPUT_H, scale);

    cv::Mat pr_img;
    float ratio_h{};
    float ratio_w{};
    ResizeImgType0(src_rgb, pr_img, 960, ratio_h, ratio_w);

    int net_in_w = pr_img.cols;
    int net_in_h = pr_img.rows;

    // method 1
    float* input_raw_data = blobFromImage(pr_img);

    // method 2
    // cv::Mat chw_image = cv::dnn::blobFromImage
    //                     (
    //                         pr_img, 1.0, // scale factor of each channel value of image
    //                         cv::Size(pr_img.colsm pr_img.rows), // spatial size for output image
    //                         cv::Scalar(127.5f), // mean
    //                         true, // swapRB: BGR to RGB
    //                         false, // crop
    //                         CV_32F // Depth of output blob. Choose CV_32F or CV_8U.
    //                     );

    
    const int i_size = 1 * 3* net_in_w*net_in_h;
    const int loc_size = 1 * 1*net_in_w*net_in_h;

    context->setBindingDimensions(i_idx, Dims4(batch_size, 3, net_in_h, net_in_w)); // for dynamic inference shape binding
    // nvinfer1::Dims dim = context->getBindingDimensions(0);

    // Create CPU buffers
    static float* h_input=nullptr, *loca=nullptr;

    Tbegin = std::chrono::steady_clock::now();

    // create input CPU lock page memory and fill data
    CHECK(cudaMallocHost((void**)&h_input, i_size*sizeof(float)));
    memcpy(h_input, input_raw_data, i_size*sizeof(float));
    // create result CPU lock page memory
    CHECK(cudaMallocHost((void**)&loca, loc_size*sizeof(float)));

    // create input GPU memory and CPU to GPU
    CHECK(cudaMalloc((void**)&buffers[i_idx], i_size*sizeof(float)));
    CHECK(cudaMemcpyAsync(buffers[i_idx], h_input, i_size*sizeof(float), cudaMemcpyHostToDevice, stream));

    // create result GPU memory
    CHECK(cudaMalloc((void**)&buffers[loc_idx], loc_size*sizeof(float)));

    context->enqueueV2((void**)buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(loca, buffers[loc_idx], loc_size*sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    Iend = std::chrono::steady_clock::now();
    float infer_time = std::chrono::duration_cast <std::chrono::milliseconds> (Iend - Tbegin).count();
    std::cout << "time : " << infer_time/1000.0 << " Sec" << std::endl;

    auto boxes = DBPostProcess(loca, src_bgr, ratio_h, ratio_w, net_in_h, net_in_w);

    Tend = std::chrono::steady_clock::now();
    float f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();

    std::cout << "time : " << f/1000.0 << " Sec" << std::endl;


    for (unsigned int i = 0; i < boxes.size(); i++) {
        auto box = boxes[i];
        std::vector<cv::Point> pts{};
        for (int j = 0; j < 4; j++)
        {   
            std::cout << ">>>>>>> x:" << box[j][0] << " | y:" << box[j][1] << std::endl;
            pts.push_back(cv::Point(box[j][0], box[j][1]));
        }
        
        cv::polylines(src_bgr, pts, true, cv::Scalar(0,0,255), 2);

        cv::imwrite("./results/"+file_names[ii], src_bgr);
    }

    cudaStreamDestroy(stream);
    cudaFreeHost(context);
    cudaFreeHost(engine);
    cudaFreeHost(runtime);

    CHECK(cudaFree(buffers[i_idx]));
    CHECK(cudaFree(buffers[loc_idx]));
    return 0;
}
