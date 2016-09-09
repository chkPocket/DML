#ifndef OWLQN_H_
#define OWLQN_H_
#include "mpi.h"
#include <iostream>
#include <algorithm>
#include <pthread.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <deque>
#include "load_data.h"
#include <glog/logging.h>

#define MASTERID 0
#define NUM 999

extern "C"{
#include <cblas.h>
}

class OWLQN{
    public:
    OWLQN(Load_Data* ld, int total_num_proc, int my_rank)
        : data(ld), num_proc(total_num_proc), rank(my_rank) {
            init();
    }

    ~OWLQN(){
        delete[] glo_w;
        delete[] glo_new_w;

        delete[] loc_z;

        delete[] loc_g;
        delete[] glo_g;
        delete[] loc_new_g;
        delete[] glo_new_g;

        delete[] glo_sub_g;

        delete[] glo_q;

        for(int i = 0; i < m; i++){
            delete[] glo_s_list[i];
            delete[] glo_y_list[i];
        }
        delete[] glo_s_list;
        delete[] glo_y_list;

        delete[] glo_alpha_list;
        delete[] glo_ro_list;
    }

    void init(){
        c = 1.0;
        glo_w = new double[data->glo_fea_dim]();
        glo_new_w = new double[data->glo_fea_dim]();
        srand(time(NULL));
        for(int i = 0; i < data->glo_fea_dim; i++) {
            glo_w[i] = 0.0;
        }

        loc_z = new double[data->loc_ins_num]();

        loc_g = new double[data->glo_fea_dim]();
        glo_g = new double[data->glo_fea_dim]();
        loc_new_g = new double[data->glo_fea_dim]();
        glo_new_g = new double[data->glo_fea_dim]();

        glo_sub_g = new double[data->glo_fea_dim]();

        glo_q = new double[data->glo_fea_dim]();

        m = 10;
        now_m = 1;
        glo_s_list = new double*[m];
        for(int i = 0; i < m; i++){
                glo_s_list[i] = new double[data->glo_fea_dim]();
                for(int j = 0; j < data->glo_fea_dim; j++){
                        glo_s_list[i][j] = glo_w[j];
                }
        }
        glo_y_list = new double*[m];
        for(int i = 0; i < m; i++){
                glo_y_list[i] = new double[data->glo_fea_dim]();
                for(int j = 0; j < data->glo_fea_dim; j++){
                        glo_y_list[i][j] = glo_g[j];
                }
        }
        glo_alpha_list = new double[data->glo_fea_dim]();
        glo_ro_list = new double[data->glo_fea_dim]();

        loc_loss = 0.0;
        glo_loss = 0.0;
        loc_new_loss = 0.0;
        glo_new_loss = 0.0;

        lambda = 1.0;
        backoff = 0.5;

        step = 10;
    }

    void calculate_z(double *w){
        size_t idx = 0;
        double val = 0;
        for(int i = 0; i < data->loc_ins_num; i++) {
            loc_z[i] = 0;
            for(int j = 0; j < data->fea_matrix[i].size(); j++) {
                idx = data->fea_matrix[i][j].idx;
                val = data->fea_matrix[i][j].val;
                loc_z[i] += w[idx] * val;
            }
        }
    }

    double sigmoid(double x){
    if(x < -30){
        return 1e-6;
    }
    else if(x > 30){
        return 1.0;
    }
    else{
        double ex = pow(2.718281828, x);
        return ex / (1.0 + ex);
    }
    }

    double calculate_loss(double *w){
        double f = 0.0, val = 0.0, wx = 0.0, single_loss = 0.0, regular_loss = 0.0;
        int index;
        memset(loc_z, 0, sizeof(double) * data->loc_ins_num);
        calculate_z(w);
        for(int i = 0; i < data->fea_matrix.size(); i++){
            wx = 0.0;
            for(int j = 0; j < data->fea_matrix[i].size(); j++){
                index = data->fea_matrix[i][j].idx;
                val = data->fea_matrix[i][j].val;
                //LOG(INFO) << *(para_w + index) << std::endl;
            }
            //LOG(INFO)<<"wx: "<<sigmoid(wx)<<std::endl;
            single_loss = data->label[i] * log(sigmoid(loc_z[i])) +
                      (1 - data->label[i]) * log(1 - sigmoid(loc_z[i]));
            f += single_loss;
        }
        for(int j = 0; j < data->fea_matrix[0].size(); j++){
            regular_loss += abs( *(w + index) );
        }
        return -f / data->fea_matrix.size() + regular_loss;
    }

    void calculate_gradient(double *w){
        double value;
        int index, single_feature_num, instance_num = data->fea_matrix.size();
        //LOG(INFO) << "process " << rank << ", instance num "
        std::cout << "process " << rank << ", instance num "
        << instance_num << std::endl;
        loc_loss = calculate_loss(w);
        for(int i = 0; i < instance_num; i++){
            single_feature_num = data->fea_matrix[i].size();
            for(int j = 0; j < single_feature_num; j++){
                index = data->fea_matrix[i][j].idx;
                value = data->fea_matrix[i][j].val;
                loc_g[index] += (sigmoid(loc_z[i]) - data->label[i]) * value;
                //DLOG(INFO) << "loc_g[" << index << "]: " << loc_g[index]
                //           << " after instance " << i + 1  << "/" << instance_num
                //           << " in rank " << rank <<std::endl << std::flush;
            }
            /*
               if(i == instance_num - 1){
               for(int index = 0; index < data->glo_fea_dim; index++)
               std::cout << "loc_g[" << index << "]: " << loc_g[index]
                     << " after instance " << i + 1  << "/" << instance_num
                     << " in rank " << rank <<std::endl << std::flush;
               }*/
        }
        /*for(int index = 0; index < data->glo_fea_dim; index++){
        glo_g[index] = loc_g[index] / instance_num;
        //std::cout << "glo_g[" << index << "]: " << glo_g[index] << " after normal "
        //          << "in rank " << rank <<std::endl << std::flush;	
        }*/
    }//end calculate_gradient

    void calculate_subgradient(){
        if(c == 0.0){
            for(int j = 0; j < data->glo_fea_dim; j++){
                *(glo_sub_g + j) = -1 * *(glo_g + j);
            }
        } else if(c != 0.0){
            for(int j = 0; j < data->glo_fea_dim; j++){
                //LOG(INFO) << *(glo_g + j) << std::endl;
                //LOG(INFO) << *(glo_w + j) << std::endl;
                if(*(glo_w + j) > 0){
                    *(glo_sub_g + j) = *(glo_g + j) + c;
                }
                else if(*(glo_w + j) < 0){
                    *(glo_sub_g + j) = *(glo_g + j) - c;
                }
                else {
                    //LOG(INFO) << *(glo_g + j) - c << std::endl;
                    if(*(glo_g + j) - c > 0){
                        *(glo_sub_g + j) = *(glo_g + j) - c;//左导数
                    } else if(*(glo_g + j) + c < 0){
                        *(glo_sub_g + j) = *(glo_g + j) + c;
                    } else {
                        *(glo_sub_g + j) = 0;
                    }
                }
                //LOG(INFO) << *(glo_sub_g + j) << std::endl;
                //LOG(INFO) << c <<std::endl;
            }
        }
        /*
           for(int i = 0; i < data->glo_fea_dim; i++){
           if(rank == 0)
           std::cout<<"glo_sub_g["<<i<<"]: "<<glo_sub_g[i]
           <<" in rank:" <<rank<<std::endl;
           }*/
    }

    void fix_dir_glo_q(){
    /*
       for(int j = 0; j < data->glo_fea_dim; j++){
    //if(rank == 0) std::cout<<"glo_q["<<j<<"]"<<glo_q[j]<<std::endl;
    if(rank == 0) std::cout<<"glo_sub_g["<<j<<"]"<<glo_sub_g[j]<<std::endl;
    }
    */
    for(int j = 0; j < data->glo_fea_dim; ++j){
        if(*(glo_q + j) * *(glo_sub_g +j) >= 0){
            *(glo_q + j) = 0.0;
        }
    }
    /*
   for(int j = 0; j < data->glo_fea_dim; ++j){
   std::cout<<"glo_q["<<j<<"]: "<<glo_q[j]<<std::endl;
   }
   */
    }

    void fix_dir_glo_new_w(){
    for(int j = 0; j < data->glo_fea_dim; j++){
        if(*(glo_new_w + j) * *(glo_w + j) >=0) *(glo_new_w + j) = 0.0;
        else *(glo_new_w + j) = *(glo_new_w + j);
    }
    }   

    void line_search(){
        MPI_Status status;
        while(true){
            if(rank == MASTERID){
                for(int j = 0; j < data->glo_fea_dim; j++){
                        //local_g equal all nodes g
                        *(glo_new_w + j) = *(glo_w + j) + lambda * *(glo_q + j);
                }
                for(int i = 1; i < num_proc; i++){
                        MPI_Send(glo_new_w, data->glo_fea_dim, MPI_DOUBLE, i,
                                        99, MPI_COMM_WORLD);
                }
            } else if(rank != MASTERID){
                MPI_Recv(glo_new_w, data->glo_fea_dim, MPI_DOUBLE, 0,
                                        99, MPI_COMM_WORLD, &status);
            }

            loc_new_loss = calculate_loss(glo_new_w);

            if(rank != MASTERID){
                    MPI_Send(&loc_new_loss, data->glo_fea_dim, MPI_FLOAT, 0,
                                    9999, MPI_COMM_WORLD);
            } else if(rank == MASTERID){
                glo_new_loss += loc_new_loss;
                for(int i = 0; i < num_proc; i++){
                        MPI_Recv(&loc_new_loss, data->glo_fea_dim, MPI_FLOAT, i,
                                        99, MPI_COMM_WORLD, &status);
                        glo_new_loss += loc_new_loss;
                }
                //LOG(INFO) << "masterid:" << MASTER_ID << std::endl;
                //LOG(INFO) << "before reduce rank:" << rank <<" loc_new_loss:"
                //          << loc_new_loss << " glo_loss:" << glo_loss
                //          << std::endl;
                //MPI_Allreduce(&loc_new_loss, &glo_new_loss, 1, MPI_DOUBLE,
                //              MPI_SUM, MPI_COMM_WORLD);
                //LOG(INFO) << "after reduce rank:" << rank << " glo_new_loss:"
                //          << glo_new_loss << " glo_loss:" << glo_loss
                //          << std::endl;
                double expected_loss = glo_loss + lambda *
                        cblas_ddot(data->glo_fea_dim, (double*)glo_q,
                                        1, (double*)glo_sub_g, 1);
                if(glo_new_loss <= expected_loss){
                    break;
                }
                lambda *= backoff;
                //LOG(INFO) << lambda << std::endl;
                if(lambda <= 1e-6) break;
                LOG(INFO) << "lambda is less than 1e-6 in line search, "
                        << "lambda value [" << lambda << " ]" << std::endl;
            }
        }
    }

    void two_loop(){
        cblas_dcopy(data->glo_fea_dim, glo_sub_g, 1, glo_q, 1);
        if(now_m > m) now_m = m;
        for(int loop = now_m-1; loop >= 0; --loop){
                glo_ro_list[loop] =
                        cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop],
                                        1, &(*glo_s_list)[loop], 1);
                glo_alpha_list[loop] =
                        cblas_ddot(data->glo_fea_dim, &(*glo_s_list)[loop],
                                        1, (double*)glo_q, 1) /
                        (glo_ro_list[loop] + 1.0);
                cblas_daxpy(data->glo_fea_dim, -1 * glo_alpha_list[loop],
                                &(*glo_y_list)[loop], 1, (double*)glo_q, 1);
        }
        if(step != 0){//if step not equal 0, scale glo_q by gamma;
                double ydoty =
                        cblas_ddot(data->glo_fea_dim, glo_s_list[now_m - 1],
                                        1, glo_y_list[now_m - 1], 1);
                float gamma = glo_ro_list[now_m - 1] / ydoty;
                cblas_dscal(data->glo_fea_dim, gamma, (double*)glo_q, 1);
        }
        for(int loop = 0; loop < now_m; ++loop){
                double beta =
                        cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop],
                                        1, (double*)glo_q, 1) /
                        (glo_ro_list[loop] + 1.0);
                cblas_daxpy(data->glo_fea_dim, glo_alpha_list[loop] - beta,
                                &(*glo_s_list)[loop], 1, (double*)glo_q, 1);
        }
    }

    void update_state(){
        //update lbfgs memory
        update_memory();//not distributed
        //update w
        std::swap(glo_w, glo_new_w);
        //update loss
        glo_loss = glo_new_loss;
        //update step count
        step++;
    }

    void update_memory(){
    //update slist
    cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_w, 1,
            (double*)glo_new_w, 1);
    cblas_dcopy(data->glo_fea_dim, (double*)glo_new_w, 1,
            (double*)glo_s_list[now_m % m], 1);
    //update ylist
    cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_g, 1,
            (double*)glo_new_g, 1);
    cblas_dcopy(data->glo_fea_dim, (double*)glo_new_g, 1,
            (double*)glo_y_list[now_m % m], 1);
    now_m++;
    } 

    bool meet_criterion(){
    if(step == 300) return true;
    return false;
    } 

    void save_model() {
        if(MASTERID == rank) {
            time_t rawtime;
            struct tm* timeinfo;
            char buffer[80];
            time(&rawtime);
            timeinfo = localtime(&rawtime);
            strftime(buffer, 80, "%Y%m%d_%H%M%S", timeinfo);
            std::string time_s = buffer;

            std::ofstream md;
            md.open("./model/lr_model_" + time_s + ".txt");
            double wi = 0.0;
            for(int i = 0; i < data->glo_fea_dim; ++i) {
                wi = glo_new_w[i];
                md << i << ':' << wi;
                if(i != data->glo_fea_dim - 1){
                    md << ' ';
                }
            }
            md.close();
        }
    }

    void owlqn(){
        for(int i = 0; i < step; i++){
            MPI_Status status;
            calculate_gradient(glo_w);
            if(rank != MASTERID){
                MPI_Send(loc_g, data->glo_fea_dim, MPI_DOUBLE, MASTERID,
                    99, MPI_COMM_WORLD);
            } else if(rank == MASTERID){
                for(int i = 1; i < num_proc; i++){
                    MPI_Recv(glo_g, data->glo_fea_dim, MPI_DOUBLE, i,
                        99, MPI_COMM_WORLD, &status);
                }
                calculate_subgradient();
                two_loop();
                fix_dir_glo_q();
            }
            line_search();
  
            //not distributed, only on master process
            fix_dir_glo_new_w();

            //not distributed, only on master process
            if(meet_criterion()){
                //not distributed, only on master process
                save_model();
                break;
            } else {
                LOG(INFO) << "process " << rank << " step " << step
                << std::endl << std::flush;
                LOG(INFO) << "======================================"
                << std::endl << std::flush;
                if(rank == MASTERID) update_state();
            }
        }
    }

    double* glo_w; //global model parameter
    private:
    Load_Data* data;

    int num_proc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world
    size_t step;

    double c; //l1 norm parameter

    double* glo_new_w; //model paramter after line search

    double* loc_z; //z = W*Xi, z is input for sigmoid(z)

    double* loc_g; //gradient of loss function compute by data on this process
    double* glo_g; //gradient of loss function compute by data on all process
    double* loc_new_g; //new local gradient
    double* glo_new_g; //new global gradient

    double* glo_sub_g; //global sub gradient

    double* glo_q; //global search direction

    int m; //number memory data we want in owlqn(lbfgs)
    int now_m; //num of memory data we got now
    double** glo_s_list; //global s list in lbfgs two loop
    double** glo_y_list; //global y list in lbfgs two loop
    double* glo_alpha_list; //global alpha list in lbfgs two loop
    double* glo_ro_list; //global ro list in lbfgs two loop

    double loc_loss; //local loss
    double glo_loss; //global loss
    double loc_new_loss; //new local loss
    double glo_new_loss; //new global loss

    double lambda; //learn rate in line search
    double backoff; //back rate in line search
};
#endif
