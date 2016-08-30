#ifndef PREDICT_H_
#define PREDICT_H_

#include <iostream>
#include <fstream>
#include "predict.h"

class Predict{
    public:
    Predict(Load_Data* load_data, int total_num_proc, int my_rank) : data(load_data), num_proc(total_num_proc), rank(my_rank){}
    ~Predict(){}

    void predict(std::vector<float> glo_w){
    std::vector<float> predict_result;
    for(int i = 0; i < data->loc_ins_num; i++) {
	float x = 0.0;
        for(int j = 0; j < data->fea_matrix[i].size(); j++) {
            int idx = data->fea_matrix[i][j].idx;
            float val = data->fea_matrix[i][j].val;
            x += glo_w[idx] * val;
        }
        if(x < -30){
            pctr = 1e-6;
        }
        else if(x > 30){
            pctr = 1.0;
        }
        else{
            double ex = pow(2.718281828, x);
            pctr = ex / (1.0 + ex);
        }
        predict_result.push_back(pctr);
    }
    for(size_t j = 0; j < predict_result.size(); j++){
        if(rank == 0){
	     std::cout<<predict_result[j]<<"\t"<<1 - data->label[j]<<"\t"<<data->label[j]<<std::endl;
	    }
    }
    }
    private:
    Load_Data* data;
    float pctr;
    //MPI process info
    int num_proc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world
};
#endif
