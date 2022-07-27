#include "NvInfer.h"
#include <iostream>
using namespace nvinfer1;

namespace myLog{

class Logger:public ILogger
{
    void log(Severity severity,const char* msg)noexcept override
    {
        if(severity<=Severity::kWARNING)
            std::cout<<msg<<std::endl;
    }
};
}