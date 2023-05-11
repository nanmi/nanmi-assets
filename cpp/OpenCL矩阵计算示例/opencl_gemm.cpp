#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << "Error: " << msg << " (code " << err << ")" << std::endl; \
        return err; \
    }

// Matrix dimensions
const int N = 1024;
const int M = 1024;
const int K = 1024;

// Load kernel source code from file
std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    cl_int err;

    // Get platform and device information
    cl_platform_id platform;
    cl_device_id device;
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERROR(err, "Failed to get platform ID");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_ERROR(err, "Failed to get device ID");

    // Create an OpenCL context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERROR(err, "Failed to create context");

    // Create a command queue
    // deprecated
    // cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    // CHECK_ERROR(err, "Failed to create command queue");

    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;  // Set desired properties
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, &properties, &err);
    CHECK_ERROR(err, "Failed to create command queue");

    // Load and compile the kernel code
    std::string kernelSource = loadKernelSource("../matrix_mul.cl");
    const char* kernelSourcePtr = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, nullptr, &err);
    CHECK_ERROR(err, "Failed to create program");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to build program");

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
    CHECK_ERROR(err, "Failed to create kernel");

    // Create input and output buffers
    std::vector<float> A(N * K, 2.0f);
    std::vector<float> B(K * M, 3.0f);
    std::vector<float> C(N * M, 0.0f);
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * A.size(), A.data(), &err);
    CHECK_ERROR(err, "Failed to create input buffer A");
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * B.size(), B.data(), &err);
    CHECK_ERROR(err, "Failed to create input buffer B");
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof(float) * C.size(), nullptr, &err);
    CHECK_ERROR(err, "Failed to create output buffer C");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    CHECK_ERROR(err, "Failed to set kernel argument 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    CHECK_ERROR(err, "Failed to set kernel argument 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    CHECK_ERROR(err, "Failed to set kernel argument 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    CHECK_ERROR(err, "Failed to set kernel argument 3");
    err = clSetKernelArg(kernel, 4, sizeof(int), &M);
    CHECK_ERROR(err, "Failed to set kernel argument 4");
    err = clSetKernelArg(kernel, 5, sizeof(int), &K);
    CHECK_ERROR(err, "Failed to set kernel argument 5");

    // Define the global and local work sizes
    size_t globalSize[2] = {N, M};
    size_t localSize[2] = {16, 16}; // Adjust the local work size based on your device

    // Enqueue the kernel for execution
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to enqueue kernel");

    // Read the result back to host memory
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);
    CHECK_ERROR(err, "Failed to read buffer C");

    // Print the result
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << C[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

