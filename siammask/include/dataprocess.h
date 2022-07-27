#include <vector>
#include <map>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "state.h"
#include <math.h>
#include "inference.h"
using namespace cv;
using namespace std;

// const int W = 50;
// const int H = 50;
// const int C = 3;

void sliceArray(float *te_im, Mat img, int dim1Start, int dim1End, int dim2Start, int dim2End, int *shape);
void sliceArray(float *te_im, float *avg_chans, int dim1Start, int dim1End, int dim2Start, int dim2End, int *shape);
float *sliceArray(float *temp, int dim1Start, int dim1End, int dim2Start, int dim2End, int *shape);
float *MatToArrayUc(Mat img);
float *MatToArrayFloat(Mat img);
int *txtToAnchorData(string path);
float *txtToHanningData(string path);
void getAvg_chans(Mat im, float *avg_chans);
Mat ArrayToMat(float *im_patch_original, int H, int W, int C);
State *siamese_init(Mat im, float *target_pos, float *target_sz);
State*  siamese_track(State *state, Mat img);
float *fourDimPermute(float *data, int *shape);
float *get_subwindow_tracking(Mat img, float *pos, float model_sz, float original_sz, float *avg_chans);
float *twoDimPermute(float *data, int *shape);
void change(float *data, int length);
float *sz(float *data, float *data1, int length);
float sz_wh(float *data);
void softmax(float *data, int *shape);

void test()
{
    float *data = new float[1000 * 1000 * 3];
    for (int i = 0; i < 1000 * 1000 * 3; i++)
    {
        data[i] = 1;
    }
    int shape[3] = {1000, 1000, 3};
    float *result = sliceArray(data, 0, 621, 139, 619, shape);
    delete data;
}

/*
data是物理上一维来存储逻辑4维的数据。是按行优先来存取的。
shape数组记录来，data逻辑上的大小：
shape[0]代表dim1的大小,shape[1]代表dim2的大小，，以此类推。

维度交换的一种理解方法：
假设first==1,sec==2,third==3,fourth=0,
那么对于python中的permute或者transpose来说。我们应该如下去做：
我们需要把上述的坐标变换，换成多次相邻两个维度之间的交换。
我们先看这个变换：0,1,2,3 -》 1,0,2,3。很明显是dim1和dim2的交换。
我们再看一个变换：0,1,2,3 -》 1,2,0,3 这个变换就不是一步到位的了。而是需要经过两次维度的交换。过程如下：
（1）0,1,2,3 -》 1,0,2,3   第一次变换中，我们交换了，dim1和dim2
（2）1,0,2,3 -》 1,2,0,3   第二次变换中，我们要基于前一次变换的结果，然后交换dim2和dim3。

对于解决这个问题的算法：为暂时使用多重指针来解决。目前只解决已确定变换要求的问题。对于在某个维度中的一般性，
和在整个维度上，所有可能变换的一般性，并没有解决。但是使用我们的思路有一点可以确定的就是：
无论是多少维度的tensor.他们任意一种坐标变换，都是多个元变换构成的。
元变换就是：任意相邻两个维度之间的一次交换。
*/
float *fourDimPermute(float *data, int *shape)
{

    /*
    第一步：使用多重指针，生成data的4D tensor.需要动态为每个维度开辟空间。
    */
    int count = 0;
    float ****data4d;
    data4d = new float ***[shape[0]];

    for (int i = 0; i < shape[0]; i++)
        *(data4d + i) = new float **[shape[1]];

    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            *(*(data4d + i) + j) = new float *[shape[2]];

    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            for (int k = 0; k < shape[2]; k++)
                *(*(*(data4d + i) + j) + k) = new float[shape[3]];

    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            for (int k = 0; k < shape[2]; k++)
                for (int l = 0; l < shape[3]; l++)
                    *(*(*(*(data4d + i) + j) + k) + l) = data[count++];

    /*
    第二步：交换
    */
    //创建first tensor用于指向我们的第一个变换后的矩阵
    float ****first;
    first = new float ***[shape[1]];

    float ***fdel1[shape[1]];
    float **fdel2[shape[1]][shape[0]];
    float *fdel3[shape[1]][shape[0]][shape[2]];

    for (int i = 0; i < shape[1]; i++)
    {

        *(first + i) = new float **[shape[0]];
        fdel1[i] = *(first + i);
    }
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[0]; j++)
        {
            *(*(first + i) + j) = new float *[shape[2]];
            fdel2[i][j] = *(*(first + i) + j);
        }

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[0]; j++)
            for (int k = 0; k < shape[2]; k++)
            {
                *(*(*(first + i) + j) + k) = new float[shape[3]];
                fdel3[i][j][k] = *(*(*(first + i) + j) + k);
            }
    //基于data4d交换一二维度。
    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
        {
            *(*(first + j) + i) = *(*(data4d + i) + j);
        }

    //创建second tensor用于指向我们的第一个变换后的矩阵
    float ****sec;
    sec = new float ***[shape[1]];

    float ***sdel1[shape[1]];
    float **sdel2[shape[1]][shape[2]];
    float *sdel3[shape[1]][shape[2]][shape[0]];

    for (int i = 0; i < shape[1]; i++)
    {
        *(sec + i) = new float **[shape[2]];
        sdel1[i] = *(sec + i);
    }
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
        {
            *(*(sec + i) + j) = new float *[shape[0]];
            sdel2[i][j] = *(*(sec + i) + j);
        }

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
            for (int k = 0; k < shape[0]; k++)
            {
                *(*(*(sec + i) + j) + k) = new float[shape[3]];
                sdel3[i][j][k] = *(*(*(sec + i) + j) + k);
            }
    //基于first交换二三维度
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[0]; j++)
            for (int k = 0; k < shape[2]; k++)
                *(*(*(sec + i) + k) + j) = *(*(*(first + i) + j) + k);

    //创建third tensor用于指向我们的第二个变换后的矩阵
    float ****third;
    third = new float ***[shape[1]];
    float ***tdel1[shape[1]];
    float **tdel2[shape[1]][shape[2]];
    float *tdel3[shape[1]][shape[2]][shape[3]];

    for (int i = 0; i < shape[1]; i++)
    {
        *(third + i) = new float **[shape[2]];
        tdel1[i] = *(third + i);
    }
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
        {
            *(*(third + i) + j) = new float *[shape[3]];
            tdel2[i][j] = *(*(third + i) + j);
        }
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
            for (int k = 0; k < shape[3]; k++)
            {
                *(*(*(third + i) + j) + k) = new float[shape[0]];
                tdel3[i][j][k] = *(*(*(third + i) + j) + k);
            }
    //基于sec交换三四维度
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
            for (int k = 0; k < shape[0]; k++)
                for (int l = 0; l < shape[3]; l++)
                    *(*(*(*(third + i) + j) + l) + k) = *(*(*(*(sec + i) + j) + k) + l);

    float *result = new float[shape[1] * shape[2] * shape[3] * shape[0]];
    count = 0;
    //cout << "[ ";
    for (int i = 0; i < shape[1]; i++)
    {
        //cout << "[ ";
        for (int j = 0; j < shape[2]; j++)
        {
            //cout << "[ ";
            for (int k = 0; k < shape[3]; k++)
            {
                //cout << "[ ";
                for (int l = 0; l < shape[0]; l++)
                {
                    result[count++] = *(*(*(*(third + i) + j) + k) + l);
                    //cout << *(*(*(*(third + i) + j) + k) + l) << ",";
                }
                //cout << " ]," << endl;
            }
            //cout << " ]," << endl;
        }
        // cout << " ]," << endl;
    }
    //cout << " ]";

    //最后释放空间。一共需要释放4个：data4d , first , sec , third.顺序是从里到外。与分配空间的时候相反。
    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            for (int k = 0; k < shape[2]; k++)
                delete *(*(*(data4d + i) + j) + k);

    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            delete *(*(data4d + i) + j);

    for (int i = 0; i < shape[0]; i++)
        delete *(data4d + i);

    delete data4d;

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[0]; j++)
            for (int k = 0; k < shape[2]; k++)
            {

                delete fdel3[i][j][k];
            }

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[0]; j++)
        {
            delete fdel2[i][j];
        }

    for (int i = 0; i < shape[1]; i++)
    {
        delete fdel1[i];
    }

    delete first;

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
            for (int k = 0; k < shape[0]; k++)
            {
                delete sdel3[i][j][k];
            }

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
        {
            delete sdel2[i][j];
        }

    for (int i = 0; i < shape[1]; i++)
    {
        delete sdel1[i];
    }

    delete sec;

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
            for (int k = 0; k < shape[3]; k++)
            {
                delete tdel3[i][j][k];
            }
    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[2]; j++)
        {
            delete tdel2[i][j];
        }
    for (int i = 0; i < shape[1]; i++)
    {
        delete tdel1[i];
    }

    delete third;

    cout << "释放完成" << endl;

    return result;
}

float *twoDimPermute(float *data, int *shape)
{

    float **a = new float *[shape[0]];
    int count = 0;
    for (int i = 0; i < shape[0]; i++)
        *(a + i) = new float[shape[1]];

    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            *(*(a + i) + j) = data[count++];

    float **b = new float *[shape[1]];
    float *t[shape[1]];
    for (int i = 0; i < shape[1]; i++)
    {
        *(b + i) = new float[shape[0]];
        t[i] = *(b + i);
    }

    for (int i = 0; i < shape[0]; i++)
        for (int j = 0; j < shape[1]; j++)
            *(*(b + j) + i) = *(*(a + i) + j);

    float *result = new float[shape[1] * shape[0]];

    count = 0;

    for (int i = 0; i < shape[1]; i++)
        for (int j = 0; j < shape[0]; j++)
            result[count++] = *(*(b + i) + j);

    for (int i = 0; i < shape[0]; i++)
        delete *(a + i);

    delete a;

    for (int i = 0; i < shape[1]; i++)
        delete t[i];

    delete b;

    return result;
}

void transport(float *source, float *result, int W, int H, int C)
{   
    int count = 0;

    vector<float> changeArr;
    for (int col = 0; col < C; col++)
    {   
        for (int arrCount = 0; arrCount < W; arrCount++)
        {
            int offset = arrCount * W;
            for (int row = 0; row < H; row++)
            {
                int index = (row + offset) * C + col;
                changeArr.push_back(source[index]);
                // cout<<"check--22"<<"  "<<count++<<endl;
            }
        }

    }

    std::copy(changeArr.begin(), changeArr.end(), result);
}

void getAvg_chans(Mat img, float *avg_chans)
{

    float chan1 = 0;
    float chan2 = 0;
    float chan3 = 0;

    for (int i = 0; i < img.rows; i++)
    {

        for (int j = 0; j < img.cols; j++) // 0 1 2 3 4 5
        {
            chan1 = float(img.at<Vec3b>(i, j).val[0]) + chan1;
            chan2 = float(img.at<Vec3b>(i, j).val[1]) + chan2;
            chan3 = float(img.at<Vec3b>(i, j).val[2]) + chan3;
        }
    }

    avg_chans[0] = chan1 / (img.rows * img.cols);
    avg_chans[1] = chan2 / (img.rows * img.cols);
    avg_chans[2] = chan3 / (img.rows * img.cols);
}

/*
target_pos和target_sz 这两个参数应该new到heap。然后会在State类析构时释放掉内存。
*/
State *siamese_init(Mat img, float *target_pos, float *target_sz)
{
    State *s = new State();
    s->setIm_h(img.rows);
    s->setIm_w(img.cols);
    TrackConfig *tc = new TrackConfig();

    float* avg_chans = new float[3];

    getAvg_chans(img, avg_chans); //该函数只供3通道使用

    //     cout<<avg_chans[0]<<endl;
    // cout<<avg_chans[1]<<endl;
    // cout<<avg_chans[2]<<endl;


    float wc_z = target_sz[0] + tc->context_amount * (target_sz[0] + target_sz[1]);
     //cout<<"wc_z:"<<wc_z<<endl;
    float hc_z = target_sz[1] + tc->context_amount * (target_sz[0] + target_sz[1]);
    //cout<<"hc_z:"<<hc_z<<endl;
    float s_z = round(sqrt(wc_z * hc_z));
    //cout<<"s_z:"<<s_z<<endl;
    // cout<<img<<endl;
    // cout<<tc->exemplar_size<<endl;
    // cout<<s_z<<endl;
    

    // z_crop是指向
   
    float *z_crop = get_subwindow_tracking(img, target_pos, tc->exemplar_size, s_z, avg_chans);
    

    string path = "/home/honsen/hanning.txt";
    string path1 = "/home/honsen/anchor.txt";
    float *window = txtToHanningData(path);
    int *anchor = txtToAnchorData(path1);

    s->setWindow(window);
    s->setTarget_pos(target_pos);
    s->setTarget_sz(target_sz);
    s->setAnchors(anchor);
    s->setZ_crop(z_crop);
    s->setAvg_chans(avg_chans);
    s->trackConfig = tc;
    /*
    在这里得到的z_crop就是以物理一维方式存放的逻辑3维的数据，格式为（3,127,127）
    */
    return s;
}

State* siamese_track(State* state,Mat img){

TrackConfig* tc = state->trackConfig;
float* z_crop = state->getZ_crop();
 
float* avg_chans = state->getAvg_chans();
// for(int i = 0;i<3;i++)
//         cout<<avg_chans[i]<<endl;
float* window = state->getWindow();

float* target_pos = state->getTarget_pos();
float* target_sz = state->getTarget_sz();

float wc_x = target_sz[1] + tc->context_amount*(target_sz[0]+target_sz[1]);
float hc_x = target_sz[0] + tc->context_amount*(target_sz[0]+target_sz[1]);
float s_x = sqrt(wc_x*hc_x);
float scale_x = tc->exemplar_size/s_x;
float d_search = (tc->instance_size-tc->exemplar_size)/2;
float pad = d_search/scale_x;
s_x = s_x + 2*pad;
float crop_box[4]={target_pos[0] - round(s_x) / 2, target_pos[1] - round(s_x) / 2, round(s_x), round(s_x)};

float* x_crop = get_subwindow_tracking(img,target_pos,tc->instance_size,round(s_x),avg_chans);
cout<<"check--"<<endl;
// for(int i = 0;i<100;i++)
//         cout<<x_crop[i+20000]<<endl;
/*
如何在C++中推理得到mask的思路：
score, delta, mask = net.track_mask(x_crop.to(device)) 这是python中的代码

score对应于rpn_pred_cls,  delta对应于rpn_pred_loc,mask对应于pred_mask.这三个值是函数track_mask的返回值
在C++中，我们需要将其大致分成三个网络。分别对应于三个返回的特征。

1.首先是rpn_pred_cls和rpn_pred_loc:
（1）首先我们需要的第一个网络是ResNet,用于初步特征提取。
第一步是得到resnet的onnx,然后生成推理文件，对我们第一帧template的crop进行推理，
得到其特征，名命为zf（以一维数组的方式存储）。

（2）然后对我们的第二帧（以及后续帧），也就是search用resnet做同样的特征提取。这里我们需要两个结果：
一个search经过resnet未经下采样的特征a. 一个是search经过下采样的特征b.特征a用于
实现FPN，达到后续的refine操作。   所以我们在这里同样需要将其在C++中分成两个网络来进行推理。

（3）使用RPN来生成两个特征，即rpn_pred_cls和rpn_pred_loc。因为有两个输出，
所以我们同样需要在C++中使用DepthCorr网络来分别推理出两个特征。需要注意的是，因为cls只有正负之分，
所以最后的输出通道数为2×anchor_num,而loc因为是四个坐标位置，所以其输出通道数为4×anchor_num。

2.然后是生成mask特征部分
（1）首先我们需要利用rpn网络部分，使用forward_corr来生成网络。使用该网络计算zf和search的互相关。
得到初步互相关特征corr_feature。因为使用，提炼过程，所以我们还需要使用refine网络，通过对search
初步特征（也就是未经过下采样的特征）和corr_feature来refine我们的mask。

总结我们需要用到的网络：
（1） resnet
 (2) 对resnet提取的特征进行下采样的部分
（3） depthCorr网络(2个)
（4） depthCorr中的部分，即其中forward_corr
（5） refine部分

即我们需要在python中生成以上6个网络的onnx。

*/

double duration;



//第一步，推理第一帧获取特征
float** firstZf = templateResNet(z_crop);
float*  zf = tedownSample(firstZf[3]);



// for(int i = 0;i<10;i++)
//             cout<<zf[i]<<endl;

//第二步，推理search区域，获得初步特征和下采样后的特征
/*
这里我们可以将resNet()和DownSample()合并到resNetAndSample()中。也可以分开使用
 */
duration = static_cast<double>(cv::getTickCount());
float** firstSearch = searchResNet(x_crop);


float* search = sedownSample(firstSearch[3]);


//第三步：使用search和zf进行互相关，分别得到cls和loc
float* clsCon_kernel = clsConv_kernel(zf);


int kershape[4] = {1,256,5,5};
int sershape[4] = {1,256,29,29};
float* clsCon_search = clsConv_search(search);



float* clsCor = Fconvd(clsCon_search,clsCon_kernel,sershape,kershape);


float* pred_cls = clsCorr(clsCor);

float* locCon_kernel = locConv_kernel(zf);
float* locCon_search = locConv_search(search);


float* locCor = Fconvd(locCon_search,locCon_kernel,sershape,kershape);


float* pred_loc = locCorr(locCor);
duration = static_cast<double>(cv::getTickCount()) - duration;   
duration /= cv::getTickFrequency(); 




cout<<"推理时间："<<duration<<"s"<<endl;
delete zf;
delete search;
delete clsCon_search;
delete clsCon_kernel;
delete locCon_kernel;
delete locCon_search;
delete clsCor;
delete locCor;
delete firstZf[0];
delete firstZf[1];
delete firstZf[2];
delete firstZf[3];
delete firstZf;
delete x_crop;





//0.08  0.14 这个是整体比较 差距0.06秒
//下面只比较F.convd这个函数。然后进行减法。C++中串行计算的卷积时间为0.027  +0.013  






// float* pred_cls = DepthCorrCls(search,zf); //该网络输出的通道数为2×anchor_num
// float* pred_loc = DepthCorrLoc(search,zf); //该网络输出的通道数为4xanchor_num

//第四步：使用DepthCorr中的forward_corr来推理zf和search。得到corr_feature
//使用refine网络来推理firstSearch和corr_feature得到pred_mask
// float* corr_feature = forward_corr(zf,search);
// float* mask = refine(firstSearch,corr_feature);

int qwe = 0;
// for(int i = 0;i<100;i++)
//             cout<<pred_loc[i]<<endl;



int scoreShape[4]={1,10,25,25};//记录score的形状
int deltaShape[4]={1,20,25,25};//记录delta形状

//decode pred_loc
float* delta = fourDimPermute(pred_loc,deltaShape);
delete pred_loc;
//delta = contiguous(delta); 请看下面的说明
//delta = view(delta,4,-1);

int viewDelta[2]={4,3125} ;//重新给delta绑定一个形状

/*
在这里需要说明的是，我们不需要实现对应于python的contiguous和view。
因为经过fourDimPermute函数以后，我们得到的数组已经是对变换结果按行优先的方式
进行存储了。所以不需要使用contiguous。然后实现view的功能，只需要对应的改下
我们数据的逻辑形状即可。即重为我们的delta或者score绑定一个新的shape即可。

*/

//decode pred_cls
float* score = fourDimPermute(pred_cls,scoreShape);
delete pred_cls;
// cout<<"测试借宿--"<<endl; 
// for(int i = 0;i<10;i++)
//             cout<<delta[i]<<endl;

// cout<<"  ---- "<<endl;

// for(int i = 0;i<10;i++)
//             cout<<score[i]<<endl;

//score = contiguous(score);  请看上面的说明
//score = view(score,2,-1);

int viewScore[2]={2,3125};//重新给score绑定一个形状

int cnt = 0;



float* newScore = twoDimPermute(score,viewScore);
delete score;

viewScore[0] = 3125;
viewScore[1] = 2;
// cout<<"准备softmax-------"<<endl;

// for(int i = 0;i<10;i++)
//             cout<<newScore[i]<<endl;
// cout<<"准备softmax-------"<<endl;
softmax(newScore,viewScore); //这里的softmax是针对dim1的。也就是为每一行进行softmax.如果是dim0,就是对每一列进行。
// for(int i = 0;i<10;i++)
//             cout<<newScore[i]<<endl;
// for(int i = 0;i<10;i++)
//             cout<<"newScore:"<<newScore[1*3125+i]<<endl;
float finalScore[viewScore[0]];
for(int i = 0;i<viewScore[0];i++)
    for(int j = 0;j<viewScore[1];j++)
        {
            if(j == 1)
            {
                finalScore[i] = newScore[cnt];
            }
            cnt++;
        }

delete newScore;
// for(int i = 0;i<10;i++)
//             cout<<"finalScore:"<<finalScore[i]<<endl;
// cout<<"完成softmax-------"<<endl;

/*
    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    第一个是计算delta,第一行，所有列的数据。计算方式是让delta第一行所有列的数据乘上anchor中每一行第三列的数据。逻辑上相当于一个点积。然后加上anchor每一行第一列的数据。

    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    第二个与第一个类似

    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    第三个是计算delta第3行，所有列的数据。 计算方式是对delta第3行，所有列的数据求个exp函数，然后乘以anchor的每一行的第3列数据。

    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]
    第四个与第三个同理

    C++实现方式：
    设计相关索引方式来对物理一维存储的delta数组和anchor数组进行访问，实现逻辑上的运算操作。
*/


//delta第一行计算
int count = 0;
float* delta1 = new float[viewDelta[1]] ;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==0)
            {
                delta1[j] = delta[count];
            }
            count++;
        }

int anShape[2]={3125,4};
int* anchor3 = new int[anShape[0]] ;
int* anchor = state->getAnchors();
count = 0;
for(int i = 0;i<anShape[0];i++)
    for(int j = 0;j<anShape[1];j++)
        {
            if(j==2)
            {
                anchor3[i] = anchor[count];
            }
            count++;
        }

int* anchor1 = new int[anShape[0]] ;
count = 0;
for(int i = 0;i<anShape[0];i++)
    for(int j = 0;j<anShape[1];j++)
        {
            if(j==0)
            {
                anchor1[i] = anchor[count];
            }
            count++;
        }

count = 0;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==0)
            {
               delta[count] = anchor3[j]*delta1[j]+anchor1[j];
            }
            count++;
        }

//delta第二行计算
float* delta2 = new float[viewDelta[1]] ;
count = 0;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==1)
            {
                delta2[j] = delta[count];
            }
            count++;
        }

int* anchor4 = new int[anShape[0]] ;
count = 0;
for(int i = 0;i<anShape[0];i++)
    for(int j = 0;j<anShape[1];j++)
        {
            if(j==3)
            {
                anchor4[i] = anchor[count];
            }
            count++;
        }

int* anchor2 = new int[anShape[0]] ;
count = 0;
for(int i = 0;i<anShape[0];i++)
    for(int j = 0;j<anShape[1];j++)
        {
            if(j==1)
            {
                anchor2[i] = anchor[count];
            }
            count++;
        }

count = 0;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==1)
            {
               delta[count] = anchor4[j]*delta2[j]+anchor2[j];
            }
            count++;
        }

//delta第三行计算。
count = 0;
float* delta3 = new float[viewDelta[1]] ;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==2)
            {
                delta3[j] = exp(delta[count]);
            }
            count++;
        }

count = 0;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==2)
            {
               delta[count] = anchor3[j]*delta3[j];
            }
            count++;
        }

//delta第四行计算。
count = 0;
float* delta4 = new float[viewDelta[1]] ;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==3)
            {
                delta4[j] = exp(delta[count]);
            }
            count++;
        }

count = 0;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==3)
            {
               delta[count] = anchor4[j]*delta4[j];
            }
            count++;
        }




float target_sz_in_crop[2] = {target_sz[0]*scale_x,target_sz[1]*scale_x};

count = 0;
float d3[viewDelta[1]];
float d4[viewDelta[1]];
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==2)
            {
                d3[j] = delta[count];
            }
            count++;
        }

count = 0;
for(int i = 0;i<viewDelta[0];i++)
    for(int j = 0;j<viewDelta[1];j++)
        {
            if(i==3)
            {
                d4[j] = delta[count];
            }
            count++;
        }



cout<<"完成准备工作-------"<<endl;

float* s_c = sz(d3,d4,viewDelta[1]);


float t2 = sz_wh(target_sz_in_crop);

for(int i = 0;i<viewDelta[1];i++)
{
    s_c[i] = s_c[i]/t2;
}

change(s_c,viewDelta[1]);  //s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz_in_crop)))

t2 = target_sz_in_crop[0]/target_sz_in_crop[1];

float r_c[viewDelta[1]];

for(int i = 0;i<viewDelta[1];i++)
{
    r_c[i] = t2/(d3[i]/d4[i]);
}
change(r_c,viewDelta[1]);




float penalty[viewDelta[1]];

for(int i = 0;i<viewDelta[1];i++)
{
    penalty[i] = exp(-(r_c[i]*s_c[i]-1)*tc->penalty_k);
}

float pscore[viewDelta[1]];
for(int i = 0;i<viewDelta[1];i++)
{
    pscore[i] = penalty[i]*finalScore[i];
}
// for(int i = 0;i<10;i++)
//             cout<<"pscore:"<<pscore[i]<<endl;
for(int i = 0;i<viewDelta[1];i++)
{
    pscore[i] = pscore[i]*(1-tc->window_influence)+window[i]*tc->window_influence;

}


//pscore数组中得分最大的索引
int best_pscore_id ;
int max = 0;

for(int i = 0 ;i<viewDelta[1];i++)
{
    if(pscore[i]>pscore[max])
    {
        max = i;
    }
}
best_pscore_id = max;

float pred_in_crop[4];
float bestbox[4];
count = 0;
for(int i = 0 ;i<viewDelta[0];i++)
    for(int j = 0 ;j<viewDelta[1];j++)
        {
            if(j==best_pscore_id)
                bestbox[i]=delta[count];
            count++;
        }

for(int i = 0 ;i<4;i++)
{
    pred_in_crop[i] = bestbox[i]/scale_x;
}



float lr = penalty[best_pscore_id]*finalScore[best_pscore_id]*tc->lr;

float res_x = pred_in_crop[0] + target_pos[0];
float res_y = pred_in_crop[1] + target_pos[1];

float res_w = target_sz[0]*(1-lr) + pred_in_crop[2]*lr;
float res_h = target_sz[1]*(1-lr) + pred_in_crop[3]*lr;

float finalPos[2] = {res_x,res_y};

float finalSz[2] = {res_w,res_h};






if(finalPos[0]>state->getIm_w())
    {
        finalPos[0]=state->getIm_w();
        if(finalPos[0]<0)
        {
            finalPos[0]=0;
        }
    }
    else{
        if(finalPos[0]<0)
        {
            finalPos[0]=0;
        }
    }
if(finalPos[1]>state->getIm_h())
    {
        finalPos[1]=state->getIm_h();
        if(finalPos[1]<0)
        {
            finalPos[1]=0;
        }
    }
    else{
         if(finalPos[1]<0)
        {
            finalPos[1]=0;
        }
    }

if(finalSz[0]>state->getIm_w())
    {
        finalSz[0]=state->getIm_w();
        if(finalSz[0]<10)
        {
            finalSz[0]=10;
        }
    }
    else{
         if(finalSz[0]<10)
        {
            finalSz[0]=10;
        }
    }
if(finalSz[1]>state->getIm_h())
    {
        finalSz[1]=state->getIm_h();
        if(finalSz[1]<10)
        {
            finalSz[1]=10;
        }
    }
    else{
         if(finalSz[1]<10)
        {
            finalSz[1]=10;
        }
    }

cout<<"预测位置："<<finalPos[0]<<"----"<<finalPos[1]<<endl;

cout<<"预测大小："<<finalSz[0]<<"----"<<finalSz[1]<<endl;
state->getTarget_pos()[0]=finalPos[0];
state->getTarget_pos()[1]=finalPos[1];

state->getTarget_sz()[0]=finalSz[0];
state->getTarget_sz()[1]=finalSz[1];


delete anchor1;
delete anchor2;
delete anchor3;
delete anchor4;
delete delta1;
delete delta2;
delete delta3;
delete delta4;
delete delta;

delete firstSearch[0];
delete firstSearch[1];
delete firstSearch[2];
delete firstSearch[3];
delete firstSearch;
delete s_c;

return state;




/*
上面是对推理得的loc和cls tensor进行解码得到最终我们预测的Pos和Size.

下面开始进行对预测mask的解码

*/

}




int *unravel_index(int index, int *shape)
{
    int count = 0;
    int *result = new int[3];
    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++)
        {
            for (int k = 0; k < shape[2]; k++)
            {
                if (count == index)
                {
                    result[0] = i;
                    result[1] = j;
                    result[2] = k;
                    return result;
                }
                count++;
            }
        }
    }

    cout << "不符合" << endl;

    return result;
}

void change(float *data, int length)
{

    for (int i = 0; i < length; i++)
    {
        if (data[i] > (1 / data[i]))
            data[i] = data[i];
        else
            data[i] = 1 / data[i];
    }
}

//该函数返回值使用完需要delete
float *sz(float *data, float *data1, int length)
{
    float pad[length];
    for (int i = 0; i < length; i++)
    {
        pad[i] = (data[i] + data1[i]) * 0.5;
    }

    float *sz2 = new float[length];
    for (int i = 0; i < length; i++)
    {
        sz2[i] = sqrt((data[i] + pad[i]) * (data1[i] + pad[i]));
    }

    return sz2;
}

float sz_wh(float *data)
{
    float pad = (data[0] + data[1]) * 0.5;
    float sz2 = (data[0] + pad) * (data[1] + pad);

    return sqrt(sz2);
}

void softmax(float *data, int *shape)
{
    //我们只需要实现针对dim1的。
    int count = 0;
    float sumExp = 0;
    for (int rows = 0; rows < shape[0]; rows++)
    {

        for (int cols = 0; cols < shape[1]; cols++)
        {
            sumExp = exp(data[count + cols]) + sumExp;
        }

        for (int cols = 0; cols < shape[1]; cols++)
        {
            data[count + cols] = exp(data[count + cols]) / sumExp;
        }

        count = count + shape[1];
        sumExp=0;
    }
}

float max(float a, float b)
{
    if (a > b)
        return a;
    else
        return b;
}

/*
该函数返回的是指向一个堆区的指针，也就是函数底部的result,其是用new开辟的，后续需要被手动delete。
*/
float *get_subwindow_tracking(Mat img, float *pos, float model_sz, float original_sz, float *avg_chans)
{
    float sz = original_sz;
    float im_sz[3];
    im_sz[0] = img.rows;
    im_sz[1] = img.cols;
    im_sz[2] = img.channels();

    float c = (original_sz + 1) / 2;
    float context_xmin = round(pos[0] - c);
    float context_xmax = context_xmin + sz - 1;
    float context_ymin = round(pos[1] - c);
    float context_ymax = context_ymin + sz - 1;

    int left_pad = int(max(0, -context_xmin));
    int top_pad = int(max(0, -context_ymin));
    int right_pad = int(max(0, context_xmax - im_sz[1] + 1));
    int bottom_pad = int(max(0, context_ymax - im_sz[0] + 1));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;
    // cout<<"context_xmin:"<<context_xmin<<endl;
    // cout<<"context_ymax:"<<context_ymax<<endl;
    // cout<<"context_ymin:"<<context_ymin<<endl;
    // cout<<"context_ymax:"<<context_ymax<<endl;

    int r, k;
    r = img.rows;
    k = img.channels();
    int col = img.cols;
    float *im_patch_original; //= new float[1000 * 1000 * 3];
    Mat im_patch;
   
    if (left_pad != 0 || top_pad != 0 || right_pad != 0 || bottom_pad != 0)
    {
         
        float *te_im = new float[(r + top_pad + bottom_pad) * (col + left_pad + right_pad) * k];
        for (int i = 0; i < (r + top_pad + bottom_pad) * (col + left_pad + right_pad) * k; i++)
            te_im[i] = 0;

        int dim1Start, dim1End, dim2Start, dim2End, dim3Start, dim3End;
        dim1Start = top_pad;
        dim1End = top_pad + r;
        dim2Start = left_pad;
        dim2End = left_pad + col;
        dim3Start = 0;
        dim3End = 3;
        int shape[] = {int(r + top_pad + bottom_pad), int(col + left_pad + right_pad), k};
        
        sliceArray(te_im, img, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 一号
        // cout<<"切片一完成"<<endl; 
        if (top_pad)
        {
            dim1Start = 0;
            dim1End = top_pad;
            dim2Start = left_pad;
            dim2End = left_pad + col;
            // cout << "---top--" << endl;  
            // for(int i = 0;i<3;i++)
            //    cout << avg_chans[i]<< endl;   
            sliceArray(te_im, avg_chans, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 二号
             
        }
        if (bottom_pad)
        {
            dim1Start = top_pad + r;
            dim1End = r + top_pad + bottom_pad;
            dim2Start = left_pad;
            dim2End = left_pad + col;
// cout << "---bottom--" << endl;
            sliceArray(te_im, avg_chans, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 二号
        }
        if (left_pad)
        {
            dim1Start = 0;
            dim1End = r + top_pad + bottom_pad;
            dim2Start = 0;
            dim2End = left_pad;
// cout << "---left_pad--" << endl;
            sliceArray(te_im, avg_chans, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 二号
        }
        if (right_pad)
        {
            dim1Start = 0;
            dim1End = r + top_pad + bottom_pad;
            dim2Start = col + left_pad;
            dim2End = col + left_pad + right_pad;
// cout << "---right--" << endl;
            sliceArray(te_im, avg_chans, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 二号
        }

        dim1Start = int(context_ymin);
        dim1End = int(context_ymax + 1);
        dim2Start = int(context_xmin);
        dim2End = int(context_xmax + 1);
// cout << "---other--" << endl;
        im_patch_original = sliceArray(te_im, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 三号

        delete te_im;
    }
    else
    {
         
        int shape[] = {int(r + top_pad + bottom_pad), int(col + left_pad + right_pad), k};
        int dim1Start = int(context_ymin);
        int dim1End = int(context_ymax + 1);
        int dim2Start = int(context_xmin);
        int dim2End = int(context_xmax + 1);
        int dim3Start = 0;
        int dim3End = 3;
        float *temp = MatToArrayUc(img);
        

        im_patch_original = sliceArray(temp, dim1Start, dim1End, dim2Start, dim2End, shape); // sliceArray 三号
       delete temp;
        //  cout << "---555555555--" << endl;
    }
    // int length = ( int(context_ymax+1) - int(context_ymin) ) * ( int(context_xmax+1)  - int(context_xmin) ) * 3;
    int H = int(context_ymax + 1) - int(context_ymin);
    int W = int(context_xmax + 1) - int(context_xmin);
    int C = 3;
    Mat im_patch_origin = ArrayToMat(im_patch_original, H, W, C);

    if (sz != model_sz)
    {

        resize(im_patch_origin, im_patch, Size(int(model_sz), int(model_sz)), 0, 0, INTER_LINEAR);
    }
    else
    {

        im_patch = ArrayToMat(im_patch_original, H, W, C);
    }

    // cout<<"col:"<<im_patch.cols<<endl;
    // cout<<"row:"<<im_patch.rows<<endl;   
    // cout<<im_patch<<endl; 

    float *init = MatToArrayFloat(im_patch);

    


    float *result = new float[int(model_sz) * int(model_sz) * C];
    
    
    transport(init, result, int(model_sz), int(model_sz), C); //变成标准格式的tensor(B,C,W,H);

    delete init;

    delete im_patch_original;
        // cout<<"+++++++++++++++"<<endl;
    return result;
}

// sliceArray 一号   这个方法的前提是img的的三个dim都必须比te_im的小
void sliceArray(float *te_im, Mat img, int dim1Start, int dim1End, int dim2Start, int dim2End, int *shape)
{

    if (shape[0] < img.rows || shape[1] < img.cols || shape[2] < img.channels())
    {
        // cout<<shape[0]<<" +++ "<<img.rows<<endl;
        // cout<<shape[1]<<" +++ "<<img.cols<<endl;
        cout << "img的任意一个维度都不能比所操作的对象大!" << endl;
        return;
    }

    int temp = dim1Start * shape[1] * shape[2];

    for (int i = 0; i < img.rows; i++)
    {
        temp += dim2Start * shape[2];
        if (temp < (dim1End * shape[1] * shape[2]))
        {

            for (int j = 0; j < img.cols; j++)
            {

                for (int k = 0; k < img.channels(); k++)
                {

                    te_im[temp++] = float(img.at<Vec3b>(i, j).val[k]);
                }
            }
        }
        temp += (shape[1] - dim2End) * shape[2];
    }
}

// sliceArray 二号
void sliceArray(float *te_im, float *avg_chans, int dim1Start, int dim1End, int dim2Start, int dim2End, int *shape)
{
    /*
    shape[0] 是行数
    shape[1] 是列数
    shape[2] 是通道数
    该shape为te_im的形状
    */

    for (int i = dim1Start; i < dim1End; i++)
    {
        int temp = i * shape[1] * shape[2];

        for (int j = dim2Start; j < dim2End; j++)
        {
            int temp1 = j * shape[2];

            for (int k = 0; k < 3; k++)
            {
                // cout<<"erhao :"<<temp + (temp1)<<endl;
                te_im[temp + (temp1++)] = avg_chans[k];
            }
        }
    }
}

// sliceArray 三号
/*
该方法使用完需要手动释放内存
*/
float *sliceArray(float *data, int dim1Start, int dim1End, int dim2Start, int dim2End, int *shape)
{

    float *temp = new float[(dim1End - dim1Start) * (dim2End - dim2Start) * 3];
    // cout<<shape[0]*shape[1]*shape[2]<<endl;
    // cout << (dim1End - dim1Start) * (dim2End - dim2Start) * 3 << endl;
    int count = 0;
    for (int i = dim1Start; i < dim1End; i++)
    {

        int t = i * shape[1] * shape[2];

        for (int j = dim2Start; j < dim2End; j++)
        {
            int t1 = j * shape[2];

            for (int k = 0; k < 3; k++)
            {

                if (count < (dim1End - dim1Start) * (dim2End - dim2Start) * 3)
                {
                    temp[count++] = data[t + (t1++)];
                    // cout<<(t+t1-1)<<endl;
                    // cout<<count<<endl;
                    
                }
                    

                //  cout<<count<<endl;
            }
        }
    }

    return temp;
}

Mat ArrayToMat(float *im_patch_original, int H, int W, int C)
{

    Mat img(H, W, CV_32FC3);
    int count = 0;
    for (int i = 0; i < img.rows; i++)
    {

        for (int j = 0; j < img.cols; j++) // 0 1 2 3 4 5
        {

            img.at<Vec3f>(i, j).val[0] = im_patch_original[count++];
            img.at<Vec3f>(i, j).val[1] = im_patch_original[count++];
            img.at<Vec3f>(i, j).val[2] = im_patch_original[count++];
        }
    }

    return img;
}

float *MatToArrayUc(Mat img)
{

    float *data = new float[img.rows * img.cols * 3];
    int count = 0;
    for (int i = 0; i < img.rows; i++)
    {

        for (int j = 0; j < img.cols; j++)
        {
            data[count++] = float(img.at<Vec3b>(i, j).val[0]);
            data[count++] = float(img.at<Vec3b>(i, j).val[1]);
            data[count++] = float(img.at<Vec3b>(i, j).val[2]);
        }
    }

    return data;
}

float *MatToArrayFloat(Mat img)
{

    float *data = new float[img.rows * img.cols * 3];
    int count = 0;
    for (int i = 0; i < img.rows; i++)
    {

        for (int j = 0; j < img.cols; j++)
        {
            data[count++] = float(img.at<Vec3f>(i, j).val[0]);
            data[count++] = float(img.at<Vec3f>(i, j).val[1]);
            data[count++] = float(img.at<Vec3f>(i, j).val[2]);
        }
    }

    return data;
}

int *txtToAnchorData(string path)
{

    ifstream ifs;
    ifs.open(path, ios::in);

    if (!ifs.is_open())
    {

        cout << "文件打开失败！" << endl;
        return 0;
    }

    char buf[1024] = {0};
    int *a = new int[3125 * 4];
    int count = 0;
    int num;

    while (ifs >> buf)
    {
        num = atoi(buf);

        a[count++] = num;
    }

    return a;
}

float *txtToHanningData(string path)
{
    ifstream ifs;
    ifs.open(path, ios::in);

    if (!ifs.is_open())
    {

        cout << "文件打开失败！" << endl;
        return 0;
    }

    char buf[1024] = {0};
    float *a = new float[3125];
    int count = 0;
    float num;

    while (ifs >> buf)
    {
        num = atof(buf);

        a[count++] = num;
    }

    return a;
}
