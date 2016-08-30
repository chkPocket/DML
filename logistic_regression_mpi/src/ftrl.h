#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "mpi.h"
#include <math.h>

class FTRL{
    public:
        FTRL(Load_Data* load_data, int total_num_proc, int my_rank) 
            : data(load_data), num_proc(total_num_proc), rank(my_rank){
            init();
        }

        ~FTRL(){}

    void init(){
        loc_w = new float[data->glo_fea_dim]();
        glo_w = new float[data->glo_fea_dim]();
        loc_g = new float[data->glo_fea_dim]();
        glo_g = new float[data->glo_fea_dim]();

        loc_z = new float[data->glo_fea_dim]();
        loc_sigma = new float[data->glo_fea_dim]();
        loc_n = new float[data->glo_fea_dim]();
    
        alpha = 1.0;
        beta = 1.0;
        lambda1 = 0.0;
        lambda2 = 1.0;
        bias = 0.1; 

        step = 1000;
        batch_size = 2;
    }

    float sigmoid(float x){
        if(x < -30) return 1e-6;
        else if(x > 30) return 1.0;
        else{
            double ex = pow(2.718281828, x);
            return ex / (1.0 + ex);
        }
    }

    void update(){
        MPI_Status status;
        for(int j = 0; j < data->glo_fea_dim; j++){//store local gradient to glo_g;
            glo_g[j] = loc_g[j];
        }
        for(int rankid = 1; rankid < num_proc; rankid++){//receive other node`s gradient and store to glo_g;
            MPI_Recv(loc_g, data->glo_fea_dim, MPI_FLOAT, rankid, 99, MPI_COMM_WORLD, &status);
            for(int j = 0; j < data->glo_fea_dim; j++){
                glo_g[j] += loc_g[j];
            }
        }
        for(int col = 0; col < data->glo_fea_dim; col++){
            loc_sigma[col] = ( sqrt (loc_n[col] + glo_g[col] * glo_g[col]) - sqrt(loc_n[col]) ) / alpha;
            loc_z[col] += glo_g[col] - loc_sigma[col] * loc_w[col];
            loc_n[col] += glo_g[col] * glo_g[col];

            if(abs(loc_z[col]) <= lambda1){
                 loc_w[col] = 0.0;
            }
            else{
                float tmpr= 0.0;
                if(loc_z[col] >= 0) tmpr = loc_z[col] - lambda1;
                else tmpr = loc_z[col] + lambda1;
                float tmpl = -1 * ( ( beta + sqrt(loc_n[col]) ) / alpha  + lambda2);
                loc_w[col] = tmpr / tmpl;
            }
        }
    }

    void ftrl(){
        MPI_Status status;
        int index = 0, row = 0; float value = 0.0, pctr = 0.0;
        for(int i = 0; i < step; i++){
            std::cout<<"step "<<i<<std::endl;
            row = i * batch_size;
	        while( (row < (i + 1) * batch_size) && (row < data->fea_matrix.size()) ){
	            float wx = bias;
	            for(int col = 0; col < data->fea_matrix[row].size(); col++){//for one instance
	  	            index = data->fea_matrix[row][col].idx;
	                value = data->fea_matrix[row][col].val;
	                wx += loc_w[index] * value;
                }
	            pctr = sigmoid(wx);
                for(int col = 0; col < data->fea_matrix[row].size(); col++){
                    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    loc_g[index] += (pctr - data->label[row]) * value;
                }
                ++row;
            } 
            if(rank != 0){//send gradient to rank 0;
                MPI_Send(loc_g, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
            }
            else if(rank == 0){
                update();
            }

            if(rank == 0){
                for(int r = 1; r < num_proc; r++){
                    MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, r, 999, MPI_COMM_WORLD);
                }
            }
            else if(rank != 0){
                MPI_Recv(glo_w, data->glo_fea_dim, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &status);
            }
            //break;
        }//end for
    }

    float* loc_w;

    private:
    Load_Data* data;
    int step;

    float* glo_w;
    float* loc_g;
    float* glo_g;
    float* loc_z;
    float* loc_sigma;
    float* loc_n;
    float alpha;
    float beta;
    float lambda1;
    float lambda2;
    int batch_size;
    float bias;
    int num_proc;
    int rank;
};
#endif
