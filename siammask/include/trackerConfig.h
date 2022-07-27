#include<iostream>

class TrackConfig{
public:
float penalty_k= 0.04;
float window_influence = 0.4;
float lr = 1.0;
float seg_thr = 0.35;
int exemplar_size = 127;
int instance_size = 255;
int total_stride = 8;
int out_size = 127;
int base_size = 8;
int score_size = (instance_size-exemplar_size)/total_stride+1+base_size;
float context_amount = 0.5;
float ratios[5] = {0.33,0.5,1,2,3};
float scales[1] = {8};
const int anchor_num = 5;
int round_dight = 0;
};