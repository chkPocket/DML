#include "ftrl.h"

FTRL::FTRL(Load_Data* ld, int total_num_proc, int my_rank) 
    : data(ld), num_proc(total_num_proc), rank(my_rank){
    init();
}
FTRL::~FTRL(){}

void FTRL::init(){
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
    
    step = 100;
    batch_size = 10;
}

float FTRL::sigmoid(float x){
    if(x < -30) return 1e-6;
    else if(x > 30) return 1.0;
    else{
        double ex = pow(2.718281828, x);
        return ex / (1.0 + ex);
    }
}

void FTRL::update_other_parameter(){
    MPI_Status status;
    for(int col = 0; col < data->glo_fea_dim; col++){
        if(rank != 0){
            MPI_Send(loc_g, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
        }
        else if(rank == 0){
            for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
                glo_g[f_idx] = loc_g[f_idx];
            }
            for(int ranknum = 1; ranknum < num_proc; ranknum++){
                MPI_Recv(loc_g, data->glo_fea_dim, MPI_FLOAT, ranknum, 99, MPI_COMM_WORLD, &status);
                for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
                    glo_g[f_idx] += loc_g[f_idx];
                }
            }

            loc_sigma[col] = (sqrt(loc_n[col] + glo_g[col] * glo_g[col]) - sqrt(loc_n[col])) / alpha;
            loc_z[col] += glo_g[col] - loc_sigma[col] * loc_w[col];
            loc_n[col] += glo_g[col] * glo_g[col];
        }
    }//end for
}

void FTRL::update_w(){
    for(int col = 0; col < data->glo_fea_dim; col++){
        if(abs(loc_z[col]) <= lambda1){
            loc_w[col] = 0.0;
        }
        else{
            float tmpr= 0.0;
            if(loc_z[col] > 0) tmpr = loc_z[col] - lambda1;
            else tmpr = loc_z[col] + lambda1;
            float tmpl = -1 * ( ( beta + sqrt(loc_n[col]) ) / alpha  + lambda2);
            loc_w[col] = tmpr / tmpl;
        }
    }
}

void FTRL::ftrl(){
    MPI_Status status;
    for(int i = 0; i < step; i++){
        int row = i * batch_size, index;
        if(rank == 0){
            update_w();
            for(int rank = 1; rank < num_proc; rank++){
                MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, rank, 99, MPI_COMM_WORLD);
            }
        }
        else if(rank != 0){
            MPI_Recv(glo_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, &status);
            for(int w_idx = 0; w_idx < data->glo_fea_dim; w_idx++){
                loc_w[w_idx] = glo_w[w_idx];
            }
        }
	while( (row < (i + 1) * batch_size) && (row < data->fea_matrix.size()) ){
	    float wx = 0.0, p = 0.0, value = 0.0;
	    for(int col = 0; col < data->fea_matrix[row].size(); col++){
	  	index = data->fea_matrix[row][col].idx;
	        value = data->fea_matrix[row][col].val;
	        wx += loc_w[index] * value;
            }
	    p = sigmoid(wx);
            loc_g[index] += (p - data->label[row]) * value;
            ++row;
        } 
	for(int col = 0; col < data->glo_fea_dim; ++col){
	    loc_g[col] /= batch_size;
	}
        update_other_parameter();     
    }//end for
}

void FTRL::run(){
    ftrl();
}
