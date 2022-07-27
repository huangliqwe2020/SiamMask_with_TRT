#include<iostream>
#include"trackerConfig.h"

class State
{
    public:

    TrackConfig* trackConfig;

    float getIm_h (){
        return this->im_h;
    }
    void setIm_h(float im_h){
        this->im_h=im_h;
    }

     float getIm_w (){
        return this->im_w;
    }
    void setIm_w(float im_w){
        this->im_w=im_w;
    }

     float* getAvg_chans (){
        return this->avg_chans;
    }
    void setAvg_chans(float* avg_chans){
        this->avg_chans=avg_chans;
    }

     float* getWindow (){
        return this->window;
    }
    void setWindow(float* window){
        this->window=window;
    }

      float* getTarget_pos (){
        return this->target_pos;
    }
    void setTarget_pos(float* target_pos){
        this->target_pos=target_pos;
    }

  float* getTarget_sz (){
        return this->target_sz;
    }
    void setTarget_sz(float* target_sz){
        this->target_sz=target_sz;
    }

    int* getAnchors (){
        return this->anchors;
    }
    void setAnchors(int* anchors){
        this->anchors=anchors;
    }

    float* getZ_crop (){
        return this->z_crop;
    }
    void setZ_crop(float* z_crop){
        this->z_crop=z_crop;
    }

    ~State(){
        if(avg_chans!=NULL)
        {
            delete avg_chans;
        }
        if(window!=NULL)
        {
            delete window;
        }
         if(target_pos!=NULL)
        {
            delete target_pos;
        }
         if(target_sz!=NULL)
        {
            delete target_sz;
        }
         if(anchors!=NULL)
        {
            delete anchors;
        }
         if(z_crop!=NULL)
        {
            delete z_crop;
        }
    }

    private:
    float im_h;
    float im_w;
    float* avg_chans;
    float* window;
    float* target_pos;
    float* target_sz;
    int* anchors;
    float* z_crop;
};