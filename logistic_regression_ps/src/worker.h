#include <iostream>
#include "load_data.h"
#include "ps.h"

namespace dmlc{
namespace linear{

//struct DataParCmd{
//    DataParCmd
//};

class Worker : public ps::App{
    public:
        Worker(const char *file_path){
            data = new Load_Data(file_path);
        }
        ~Worker(){
            delete data;
        } 
        virtual void ProcessRequest(ps::Message* request){
            Process();
        }
        float sigmoid(float x){
            if(x < -30) return 1e-6;
            else if(x > 30) return 1.0;
            else{
                double ex = pow(2.718281828, x);
                return ex / (1.0 + ex);
            }
        }
        void Process(){
            for(int i = 0; i < step; i++){
                data->load_data_minibatch(1000);
                std::vector<float> w;
                std::vector<float> g;
                std::vector<ps::Key> keys;
                std::vector<float> values;
                for(int i = 0; i < data->fea_matrix.size(); i++){
                    keys.clear(); values.clear();
                    float wx = bias;
                    for(int j = 0; j < data->fea_matrix[i].size(); j++){
                        long int index = data->fea_matrix[i][j].idx;
                        keys.push_back(index);
                        float value = data->fea_matrix[i][j].val;
                        values.push_back(value);
                    }
                    kv_.Wait(kv_.Pull(keys, &w));
                    for(int j = 0; j < w.size(); j++){
                        wx += w[j] * values[j];
                    }
                    float pctr = sigmoid(wx);
                    g.resize(keys.size());
                    for(int j = 0; j < keys.size(); j++){
                        g[j] += (pctr - data->label[i]) * values[j];
                    }
                }//end for
            }
        }
	
    Load_Data *data;
    float alpha = 1.0;
    float beta = 1.0;
    float lambda1 = 0.0;
    float lambda2 = 1.0;
    float bias = 0.1;
    int step = 1000;
    ps::KVWorker<float> kv_;
};


}
}
