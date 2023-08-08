#include <iostream>
#include <string>
#include <fstream> // file read/write
#include <sstream>
#include <vector>
#include <chrono>
#include <map>   // dict
#include <unistd.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"


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

// the input / output size and blobs
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";
static const int INPUT_W = 960;
static const int INPUT_H = 640;
static const int INPUT_SIZE = 3*INPUT_W*INPUT_H;
static const int OUTPUT_SIZE = INPUT_W*INPUT_H;



static Logger gLogger;

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
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
            std::string cur_file_name(p_dir_name);
            cur_file_name += "/";
            cur_file_name += p_file->d_name;
            // std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_W, INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

float* blobFromImage(cv::Mat& img){
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

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
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return blob;
}

// Original
cv::Mat postprocess_decode(float* feat_blob)
{
    cv::Mat output_mask(INPUT_H, INPUT_W, CV_32FC1, feat_blob);
    output_mask.convertTo(output_mask, CV_8UC3, 255);
    return output_mask;
}

// Cover
cv::Mat postprocess_decode(float* feat_blob, cv::Mat src)
{
    cv::Mat dst, temp, mask_img;
    cv::Mat mask(INPUT_H, INPUT_W, CV_32FC1, feat_blob);
    mask.convertTo(mask, CV_8UC3, 10);
    cv::Mat img1(src.size(), CV_8UC3);
    cv::resize(mask, img1, src.size());
    
    src.copyTo(dst, img1);
    float alpha = 0.7f;
    cv::addWeighted(dst, alpha, src, 1 - alpha, 0, src);
    
    return src;
}


// -------------------- do inference -----------------------------------------
void doInference(IExecutionContext& context, float* input, float* output, const int output_size, const int input_shape) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    // int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], input_shape * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, input_shape * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char const *argv[])
{
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[1]) == "-d") {
        const std::string engine_file_path {argv[2]};
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
        std::cerr << "./unet -d [.engine] ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    const std::string input_image_path {argv[3]};

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[3], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    // auto out_dims = engine->getBindingDimensions(1);

    int input_size = INPUT_SIZE;
    int output_size = OUTPUT_SIZE;
    static float* prob = new float[output_size];

    float* blob = nullptr;

    for (unsigned int i = 0; i < file_names.size(); i++)
    {
        std::string input_image = file_names[i];
        cv::Mat img = cv::imread(input_image);
        // int img_w = img.cols;
        // int img_h = img.rows;
        // cv::Mat pr_img = static_resize(img);

        cv::Mat pr_img(cv::Size(INPUT_W, INPUT_H), CV_8UC3);
        
        cv::resize(img, pr_img, cv::Size(INPUT_W, INPUT_H));
        blob = blobFromImage(pr_img);

        // run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, blob, prob, output_size, input_size);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // decode
        cv::Mat mask = postprocess_decode(prob, img);
        
        cv::imwrite("./r_"+std::to_string(i)+".jpg", mask);
    }
    
    // delete the pointer to the float
    delete blob;
    // destroy the engine
    cudaFreeHost(context);
    cudaFreeHost(engine);
    cudaFreeHost(runtime);

    return 0;
}
