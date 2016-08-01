#include <iostream>

namespace dmlc{
namespace linear{

  class Scheduler : public ps::App{
    public:
        Scheduler(){}
        ~Scheduler(){}

	virtual void ProcessResponse(ps::Message* response) { }
	virtual bool Run(){
	    std::cout<<"Connected "<<ps::NodeInfo::NumServers()<<" servers and "<<ps::NodeInfo::NumWorkers()<<" workers"<<std::endl;
	}
  };

}//end linear
}//end dmlc
