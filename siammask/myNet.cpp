
#include "dataprocess.h"


int createEngine (const char* modelFile,const char* modelName){
 myLog::Logger logger;
    //创建builder
    IBuilder* builder = createInferBuilder(logger);

    //创建网络定义
    uint32_t flag = 1U<<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //kEXPLICIT_BATCH这个flag是为了导入使用ONNX解析器的模型
    INetworkDefinition* network = builder->createNetworkV2(flag);


    IParser* parser = createParser(*network,logger);

    parser->parseFromFile(modelFile,int(ILogger::Severity::kWARNING));
    for(int32_t i=0;i<parser->getNbErrors();++i)
    {
        std::cout<<parser->getError(i)->desc()<<std::endl;
    }
     ITensor* data = network->getInput(0);
     auto dims = data->getDimensions();
     auto dim1 = dims.d[0];
     auto dim2 = dims.d[1];
     auto dim3 = dims.d[2];
     auto dim4 = dims.d[3];
     auto num = dims.nbDims;
    const char* name = data->getName();
    int net_num_input = network->getNbInputs();
    cout<<net_num_input<<endl;
    cout<<num<<endl;
    cout<<dim1<<endl;
    cout<<dim2<<endl;
    cout<<dim3<<endl;
    cout<<dim4<<endl;


    IBuilderConfig* config = builder->createBuilderConfig();
    builder->setMaxBatchSize(64);
    config->setMaxWorkspaceSize(1U<<20);

    auto profile = builder->createOptimizationProfile();
    for(int i = 0; i < net_num_input; ++i){
			auto input = network->getInput(i);
			auto input_dims = input->getDimensions();
			input_dims.d[0] = 1;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
			input_dims.d[0] = 64;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
		}
    config->addOptimizationProfile(profile);
    IHostMemory* serializedModel = builder->buildSerializedNetwork(*network,*config);
    delete parser;

    delete network;

    delete config;

    delete builder;

    assert(serializedModel != nullptr);

    std::ofstream p(modelName, std::ios::binary);
    if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    serializedModel->destroy();

    std::cout<<"success------"<<std::endl;

    return 0;
}


void test1(){

float data[20*25*25];
for(int i = 0;i<20*25*25;i++)
    {
    data[i] = i;
    }
int shape[4]={1,20,20,25};

fourDimPermute(data,shape);

}

int main(int argc,char** argv){

    
    //  createEngine ("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/onnx/locConvKer.onnx","/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/locConvKer.engine");
    //  createEngine ("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/onnx/locConvSea.onnx","/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/locConvSea.engine");
    // createEngine ("/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/onnx/tedownsample.onnx","/media/honsen/MyProj/myProject/tensorrtx-master/alexnet/build/engineFile/tedownsample.engine");
    // Mat img = imread("007.png");
    // float* data = cv2data(img);
    // float result[50*50*3];
  
    // transport(data,result,50,50,3);

    //  test();


// for(int i = 0;i<10000;i++)
// {
//     test1();
// }

    Mat imgs[70];
    string s[70];
     for(int i = 0;i<70;i++)
    {
        if(i<10)
        {
            s[i] = "img/0000"+to_string(i)+".jpg";
            imgs[i]=imread(s[i]);
        }
        
        else
        {
            s[i] = "img/000"+to_string(i)+".jpg";
            imgs[i]=imread(s[i]);
        }

    }

    State* state;
    float target_pos[]={388,235};
    float target_sz[] = {196,270};
    for(int i = 0;i<70;i++)
    {
        if(i<1)
        {
            state = siamese_init(imgs[i],target_pos,target_sz);
        }
        else{
            state = siamese_track(state,imgs[i]);
            float* pos = state->getTarget_pos();
            float* size = state->getTarget_sz();
            rectangle(imgs[i],Rect(int(pos[0]-size[0]/2),int(pos[1]-size[1]/2),size[0],size[1]),Scalar(255,255,255),1);
            imshow("result",imgs[i]);
            waitKey(1);
        }
        cout<<i<<endl;
    }
    
    
    //execInference(data);
    // createEngine ();

  

    
    
    return 0;
}

