#include<iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;
float* cv2data(Mat b){
    
    Mat img;
    resize(b,img,Size(50,50),0,0,INTER_LINEAR);
    cvtColor(img,img,COLOR_BGR2RGB);
    // mean = (223, 235, 237)  # RGB


    float* data = new float[50*50*3]; 
    int count=0;
    // cout<<first<<endl;
    cout<<img.rows<<endl;
    cout<<img.cols<<endl;
     for(int i = 0 ;i<img.rows;i++)
    {
        
        for(int j = 0;j<img.cols;j++)    // 0 1 2 3 4 5 
        {
            data[count++]=float(img.at<Vec3b>(i,j).val[0])-223;
            //img.at<Vec3b>(i,j).val[0]=(uchar)(float(img.at<Vec3b>(i,j).val[0])-223);
            data[count++]=float(img.at<Vec3b>(i,j).val[1])-235;
            //img.at<Vec3b>(i,j).val[1]=(uchar)(float(img.at<Vec3b>(i,j).val[1])-235);
            data[count++]=float(img.at<Vec3b>(i,j).val[2])-237;
            //img.at<Vec3b>(i,j).val[2]=(uchar)(float(img.at<Vec3b>(i,j).val[2])-237);
        }
    }

    //   for(int i = 0 ;i<img.rows;i++)
    // {
        
    //     for(int j = 0;j<img.cols;j++)    // 0 1 2 3 4 5 
    //     {
    //         //Vec3b* ptr = b.ptr<Vec3b>(i,j);
    //         for(int k = 0;k<img.channels();k++){
                
    //             data[count]=float(img.at<Vec3b>(i,j).val[k]);
    //             count++;
    //         }
            
    //     }
       
    // }
    
    for(int i=0;i<3;i++)
    {
        cout<<data[i]<<endl;
    }

    return data;
}