#include "NvInfer.h"
#include "myLogging.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include "cuda_runtime_api.h"
#include <map>
#include <chrono>
#include "cv2data.h"
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;
using namespace cv;
void doResNetInference(IExecutionContext &context, float *input, float **output, int batchSize);
float *Fconvd(float *data, float *kernel, int *shape, int *shape1);
float ***upsample(float *data, int *shape, int size);
float *pad(float *data, int *shape, int *padNum);

void doResNetInference(IExecutionContext &context, float *input, float **output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 5);
    void *buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("first");
    const int outputIndex2 = engine.getBindingIndex("sec");
    const int outputIndex3 = engine.getBindingIndex("third");
    const int outputIndex4 = engine.getBindingIndex("fourth");

    context.setBindingDimensions(inputIndex, Dims4(1, 3, 127, 127));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * 127 * 127 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 64 * 61 * 61 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * 256 * 31 * 31 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex3], batchSize * 512 * 15 * 15 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex4], batchSize * 1024 * 15 * 15 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * 127 * 127 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(*output, buffers[outputIndex1], batchSize * 64 * 61 * 61 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(*(output + 1), buffers[outputIndex2], batchSize * 256 * 31 * 31 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(*(output + 2), buffers[outputIndex3], batchSize * 512 * 15 * 15 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(*(output + 3), buffers[outputIndex4], batchSize * 1024 * 15 * 15 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    CHECK(cudaFree(buffers[outputIndex3]));
    CHECK(cudaFree(buffers[outputIndex4]));
}


void dosearResNetInference(IExecutionContext &context, float *input, float **output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 5);
    void *buffers[5];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("first");
    const int outputIndex2 = engine.getBindingIndex("sec");
    const int outputIndex3 = engine.getBindingIndex("third");
    const int outputIndex4 = engine.getBindingIndex("fourth");

    context.setBindingDimensions(inputIndex, Dims4(1, 3, 255, 255));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * 255 * 255 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 64 * 125 * 125 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * 256 * 63 * 63 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex3], batchSize * 512 * 31 * 31 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex4], batchSize * 1024 * 31 * 31 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * 255 * 255 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(*output, buffers[outputIndex1], batchSize * 64 * 125 * 125 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(*(output + 1), buffers[outputIndex2], batchSize * 256 * 63 * 63 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(*(output + 2), buffers[outputIndex3], batchSize * 512 * 31 * 31 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(*(output + 3), buffers[outputIndex4], batchSize * 1024 * 31 * 31 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
    CHECK(cudaFree(buffers[outputIndex3]));
    CHECK(cudaFree(buffers[outputIndex4]));
}


float **execSearResNetInference(float *data)
{
    
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/seResnet.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
     
    // Run inference
    float **prob = new float *[4];
    *prob = new float[64 * 125 * 125];
    *(prob + 1) = new float[256 * 63 * 63];
    *(prob + 2) = new float[512 * 31 * 31];
    *(prob + 3) = new float[1024 * 31 * 31];
    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    dosearResNetInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount()) - duration;   
    duration /= cv::getTickFrequency(); 
    cout<<"search推理时间："<<duration<<"s"<<endl;
    //}
    
    delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
   
    return prob;
}

float **execResNetInference(float *data)
{
    
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/resnet.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float **prob = new float *[4];
    *prob = new float[64 * 61 * 61];
    *(prob + 1) = new float[256 * 31 * 31];
    *(prob + 2) = new float[512 * 15 * 15];
    *(prob + 3) = new float[1024 * 15 * 15];
    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    
    doResNetInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
     
    //}
    delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}

void doDownSampleInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 1024, 15, 15));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 1024 * 15 * 15 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 256 * 7 * 7 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 1024 * 15 * 15 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 256 * 7 * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execDownSampleInference(float *data)
{
    
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/tedownsample.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[256 * 7 * 7];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    doDownSampleInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    //}
    
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}





void doSeDownSampleInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 1024, 31, 31));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 1024 * 31 * 31 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 256 * 31 * 31 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 1024 * 31 * 31 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 256 * 31 * 31 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execSeDownSampleInference(float *data)
{
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/sedownsample.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[256 * 31 * 31];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doSeDownSampleInference(*context, data, prob, 1);
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 
    cout<<"searchdownsample推理时间："<<duration<<"s"<<endl;
    auto end = std::chrono::system_clock::now();
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}




void doKernelInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("template");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 256, 7, 7));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 256 * 7 * 7 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 256 * 5 * 5 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 256 * 7 * 7 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 256 * 5 * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execKernelInference(float *data)
{
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/conv_kernel.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[256 * 5 * 5];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doKernelInference(*context, data, prob, 1);
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 
cout<<"cls分类卷积核推理时间："<<duration<<"s"<<endl;
    auto end = std::chrono::system_clock::now();
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}




void doLocKernelInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 256, 7, 7));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 256 * 7 * 7 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 256 * 5 * 5 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 256 * 7 * 7 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 256 * 5 * 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *locExecKernelInference(float *data)
{
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/locConvKer.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[256 * 5 * 5];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doLocKernelInference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 
cout<<"定位卷积推理时间："<<duration<<"s"<<endl;
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}



void doSearchInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("search");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 256, 31, 31));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 256 * 31 * 31 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 256 * 29 * 29 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 256 * 31 * 31 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 256 * 29 * 29 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execSearchInference(float *data)
{
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/conv_search.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[256 * 29 * 29];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doSearchInference(*context, data, prob, 1);
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 
cout<<"分类search推理时间："<<duration<<"s"<<endl;
    auto end = std::chrono::system_clock::now();
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}






void doLocSearchInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 256, 31, 31));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 256 * 31 * 31 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 256 * 29 * 29 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 256 * 31 * 31 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 256 * 29 * 29 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *locExecSearchInference(float *data)
{
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/locConvSea.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[256 * 29 * 29];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doLocSearchInference(*context, data, prob, 1);
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency();
cout<<"定位search推理时间："<<duration<<"s"<<endl;
    auto end = std::chrono::system_clock::now();
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}






void doLocInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 256, 25, 25));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 256 * 25 * 25 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 20 * 25 * 25 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 256 * 25 * 25 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 20 * 25 * 25 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execLocInference(float *data)
{
    double duration;
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/locCorr.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[20 * 25 * 25];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doLocInference(*context, data, prob, 1);
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 
cout<<"定位head推理时间："<<duration<<"s"<<endl;
    auto end = std::chrono::system_clock::now();
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}

void doClsInference(IExecutionContext &context, float *input, float *output, int batchSize)
{
    const ICudaEngine &engine = context.getEngine();


    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");
    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, 256, 25, 25));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 256 * 25 * 25 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * 10 * 25 * 25 * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 256 * 25 * 25 * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * 10 * 25 * 25 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execClsInference(float *data)
{
 double duration;   
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/clsCorr.engine", std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据
    
    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    // cout<<"------"<<engine->getBindingIndex("input")<<"----------"<<endl;
    assert(engine != nullptr);
    
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
   
    // Run inference
    float *prob = new float[10 * 25 * 25];

    // for (int i = 0; i < 100; i++) {
    auto start = std::chrono::system_clock::now();
    duration = static_cast<double>(cv::getTickCount());
    doClsInference(*context, data, prob, 1);
    duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 
cout<<"分类head推理时间："<<duration<<"s"<<endl;
    auto end = std::chrono::system_clock::now();
    //}
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}

// resnet和downsample

// float* firstSearch = resNet(x_crop);
float **templateResNet(float *data)
{
    float **output = execResNetInference(data);

    return output;
}

float **searchResNet(float *data)
{
    float **output = execSearResNetInference(data);

    return output;
}

// float* search = DownSample(firstSearch);
float *tedownSample(float *data)
{
    float *output = execDownSampleInference(data);
    return output;
}


float *sedownSample(float *data)
{
    float *output = execSeDownSampleInference(data);
    return output;
}


// depthCorr

// kernel---template
float *clsConv_kernel(float *data)
{
    float *output = execKernelInference(data);
    return output;
}

// search
float *clsConv_search(float *data)
{
    float *output = execSearchInference(data);
    return output;
}



float *locConv_kernel(float *data)
{
    float *output = locExecKernelInference(data);
    return output;
}

// search
float *locConv_search(float *data)
{
    float *output = locExecSearchInference(data);
    return output;
}


//该函数实现的是分组卷积。
float *Fconvd(float *data, float *kernel, int *shape, int *shape1)
{
    // shape是data的形状，shape1是kernel的形状
    //我们的形状以torch为标准：（B，C，H，W）

    //根据代码要求，需要使用view修改数据逻辑上的排布
    shape1[0] = shape[1];
    shape1[1] = 1;
    int h = shape[2] - shape1[2] + 1;
    int w = shape[3] - shape1[3] + 1;

    float *output = new float[shape1[1] * shape1[0] * h * w];
    int t = 0;
    int t1 = 0; // 记录是否计算完一个kernel。当t1等于kernel的大小时，将t1置0,然后横向或者竖向移动kernel.
    int rowStride = 0;
    int colStride = 0;
    int count = 0;
    float temp = 0;
    int groups = 0; //记录第几组卷积操作
    int flag = 0;
    for (int i = 0; i < shape[1]; i++)
    {

        flag = 1;
        while (flag)
        {

            t1 = i * shape1[2] * shape1[3]; // 3
            temp = 0;

            for (int j = 0; j < shape[2]; j++) 
            {
                if (j < shape1[2])
                {

                    t = (rowStride + j) * shape[3] + i * shape[2] * shape[3]; //定位到第J行。然后加上步长。
                    // cout << j << "---------" << endl;
                    t = t + colStride;
                    for (int k = 0; k < shape[3]; k++)
                    {

                        if (k < shape1[3])
                        {
                            temp = temp + data[t] * kernel[t1]; //当k小于kernel的列数时，
                            // cout<<"data:"<<data[t]<<endl;
                            // cout<<"kernel:"<<kernel[t1]<<endl;
                            t1++;
                        }
                        t = t + 1;
                    }
                    
                }
                else
                {
                    if (colStride == w - 1 && rowStride == h - 1)
                    {
                        flag = 0;
                    }

                    if (colStride < w - 1)
                    {
                        colStride++;
                    }
                    else if (rowStride < h - 1)
                    {
                        colStride = 0;
                        rowStride++;
                    }
                    output[count++] = temp;
                    // cout<<"temp:"<<temp<<endl;
                    // cout<<"colStride:"<<colStride<<endl;
                    // cout<<"rowStride:"<<rowStride<<endl;
                    // cout<<"第"<<count<<"个output"<<endl;
                    break;
                }
            }
        }
        colStride = 0;
        rowStride = 0;
    }

    return output;
}

//上采样函数，使用最邻近插值法。只对最里面两个维度进行插值。
float ***upsample(float *data, int *shape, int size)
{
    float ***temp = new float **[shape[0]];
    for (int i = 0; i < shape[0]; i++)
    {
        *(temp + i) = new float *[size];
    }
    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < size; j++)
            *(*(temp + i) + j) = new float[size];

    int k1;
    int j1;
    int countk;
    int countj;
    int flagk;
    int flagj;
    for (int i = 0; i < shape[0]; i++)
    {
        flagj = 0;
        countj = 0;
        for (int j = 0; j < shape[1]; j++)
        {

            if (size % shape[1] != 0)
            {

                if (j % 2 == 0 && flagj < (size % shape[1]))
                {

                    flagj++;
                    for (j1 = 0; j1 < size / shape[1] + 1; j1++)
                    {
                        countk = 0;
                        flagk = 0;
                        for (int k = 0; k < shape[2]; k++)
                        {

                            if (size % shape[2] != 0)
                            {

                                if (k % 2 == 0 && flagk < (size % shape[2]))
                                {
                                    for (k1 = 0; k1 < (size / shape[2] + 1); k1++)
                                    {

                                        *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                        // cout<<countk<<endl; 12345  11 2 33 4 5
                                        countk++;
                                    }
                                    flagk++;
                                }

                                else if (k % 2 != 0 || flagk >= (size % shape[2]))
                                {
                                    for (k1 = 0; k1 < (size / shape[2]); k1++)
                                    {
                                        *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                        // cout<<countk<<endl;
                                        countk++;
                                    }
                                }
                            }

                            else
                            {
                                for (k1 = 0; k1 < (size / shape[2]); k1++)
                                {
                                    *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                    countk++;
                                }
                            }
                        }

                        countj++;
                    }
                }
                else if (j % 2 != 0 || flagj >= (size % shape[1]))
                {

                    for (j1 = 0; j1 < size / shape[1]; j1++)
                    {
                        countk = 0;
                        flagk = 0;
                        for (int k = 0; k < shape[2]; k++)
                        {
                            if (size % shape[2] != 0)
                            {
                                if (k % 2 == 0 && flagk < size % shape[2])
                                {
                                    for (k1 = 0; k1 < (size / shape[2] + 1); k1++)
                                    {

                                        *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                        countk++;
                                    }
                                    flagk++;
                                }

                                else if (k % 2 != 0 || flagk >= (size % shape[2]))
                                {
                                    for (k1 = 0; k1 < (size / shape[2]); k1++)
                                    {

                                        *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                        countk++;
                                    }
                                }
                            }

                            else
                            {
                                for (k1 = 0; k1 < (size / shape[2]); k1++)
                                {
                                    *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                    countk++;
                                }
                            }
                        }

                        countj++;
                    }
                }
            }

            else
            {

                for (j1 = 0; j1 < size / shape[1]; j1++)
                {
                    countk = 0;
                    flagk = 0;
                    for (int k = 0; k < shape[2]; k++)
                    {
                        if (size % shape[2] != 0)
                        {
                            if (k % 2 == 0 && flagk < size % shape[2])
                            {
                                for (k1 = 0; k1 < (size / shape[2] + 1); k1++)
                                {
                                    *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                    countk++;
                                }
                                flagk++;
                            }

                            else if (k % 2 != 0 || flagk >= (size % shape[2]))
                            {
                                for (k1 = 0; k1 < (size / shape[2]); k1++)
                                {
                                    *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                    countk++;
                                }
                            }
                        }

                        else
                        {
                            for (k1 = 0; k1 < (size / shape[2]); k1++)
                            {
                                *(*(*(temp + i) + countj) + countk) = data[i * shape[1] * shape[2] + j * shape[2] + k];
                                countk++;
                            }
                        }
                    }

                    countj++;
                }
            }
        }
    }

    //     cout<<"-----"<<endl;
    // for(int i = 0 ;i<shape[0];i++)
    //     {
    //         cout<<"[";
    //         for(int j = 0 ;j<shape[1];j++)
    //         {
    //             cout<<"[";
    //             for(int k = 0 ;k<shape[2];k++)
    //                 cout<<data[i*shape[1]*shape[2]+j*shape[2]+k]<<",";
    //             cout<<"],"<<endl;
    //         }

    //         cout<<"],"<<endl;
    //     }
    //     for(int i = 0 ;i<shape[0];i++)
    //     {
    //         cout<<"[";
    //         for(int j = 0 ;j<size;j++)
    //         {
    //             cout<<"[";
    //             for(int k = 0 ;k<size;k++)
    //                 cout<<*(*(*(temp+i)+j)+k)<<",";
    //             cout<<"],"<<endl;
    //         }

    //         cout<<"],"<<endl;
    //     }

    return temp;
}

//输入data逻辑上的格式要求为（B，C，H，W),本函数只实现最里面两层,且pad的数值为0
float *pad(float *data, int *shape, int *padNum)
{
    int d1, d2, d3, d4;
    d1 = shape[0];
    d2 = shape[1];
    d3 = shape[2] + padNum[0] * 2;
    d4 = shape[3] + padNum[1] * 2;
    float *newData = new float[d1 * d2 * d3 * d4];

    for (int i = 0; i < d2 * d3 * d4; i++)
    {
        newData[i] = 0;
    }
    int count = 0;
    int t;
    int t1;
    for (int i = 0; i < shape[1]; i++)
    {

        for (int j = 0; j < shape[2]; j++)
        {

            for (int k = 0; k < shape[3]; k++)
            {

                newData[i * d3 * d4 + padNum[1] * d4 + j * d4 + padNum[1] + k] = data[k];
            }
        }
    }

    return newData;
}

float *clsCorr(float *data)
{
    float *output = execClsInference(data);
    return output;
}

float *locCorr(float *data)
{
    float *output = execLocInference(data);
    return output;
}

void doRefineInference(IExecutionContext &context, float *input, float *output, int batchSize, int *shape, int *shape1)
{
    const ICudaEngine &engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex("input");

    const int outputIndex1 = engine.getBindingIndex("output");

    context.setBindingDimensions(inputIndex, Dims4(1, shape[0], shape[1], shape[2]));

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * shape[0] * shape[1] * shape[2] * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * shape1[0] * shape1[1] * shape1[2] * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    context.setOptimizationProfileAsync(0, stream);
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * shape[0] * shape[1] * shape[2] * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex1], batchSize * shape1[0] * shape1[1] * shape1[2] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

float *execRefineInference(float *data, int *shape, int *shape1, const char *path)
{
    myLog::Logger logger;
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // data是推理的输入，以一维数组的方式存储3维数据

    IRuntime *runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float *prob = new float[shape1[0] * shape1[1] * shape1[2]];

    for (int i = 0; i < 10; i++)
    {
        auto start = std::chrono::system_clock::now();
        double duration;

        duration = static_cast<double>(cv::getTickCount());
        doRefineInference(*context, data, prob, 1, shape, shape1);

        duration = static_cast<double>(cv::getTickCount()) - duration;

        duration /= cv::getTickFrequency(); // 运行时间，ms为单位
        //  auto end = std::chrono::system_clock::now();
    }
delete trtModelStream;
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return prob;
}

// refine部分推理

float *OCRinfer(float *data)
{

    int shape[3] = {3, 50, 50};
    int shape1[3] = {1, 1, 400};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/myNet.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *deconvLayer(float *data)
{

    int shape[3] = {256, 1, 1};
    int shape1[3] = {32, 15, 15};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/deconv.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *h2Layer(float *data)
{

    int shape[3] = {32, 15, 15};
    int shape1[3] = {32, 15, 15};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/h2.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *v2Layer(float *data)
{

    int shape[3] = {512, 15, 15};
    int shape1[3] = {32, 15, 15};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/v2.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *h1Layer(float *data)
{

    int shape[3] = {16, 31, 31};
    int shape1[3] = {16, 31, 31};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/h1.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *v1Layer(float *data)
{

    int shape[3] = {256, 31, 31};
    int shape1[3] = {16, 31, 31};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/v1.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *h0Layer(float *data)
{

    int shape[3] = {4, 61, 61};
    int shape1[3] = {4, 61, 61};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/h0.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}

float *v0Layer(float *data)
{

    int shape[3] = {64, 61, 61};
    int shape1[3] = {4, 61, 61};
    const char *path = "/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/v0.engine";
    float *output = execRefineInference(data, shape, shape1, path);
    return output;
}