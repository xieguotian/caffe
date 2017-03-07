#ifndef CAFFE_TRAIN_MANAGER_H_
#define CAFFE_TRAIN_MANAGER_H_

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

namespace caffe{
	template<typename Dtype>
	class TrainManager {
	public:
		inline TrainManager(){ solver_.reset(); }
		inline TrainManager(string solver_proto, string gpu_ids = "", string snapshot = "", string weights = "",string log_name=""){ solver_.reset(); Init(solver_proto, gpu_ids, snapshot, weights,log_name); }
		shared_ptr<Net<Dtype>> Init(string sover_proto, string gpu_ids = "", string snapshot = "", string weights = "", string log_name="");
		shared_ptr<Net<Dtype>> Train(int iterations, shared_ptr<Net<Dtype>> net=NULL);
		inline shared_ptr<Net<Dtype>> net(){ return solver_->net(); }
		inline vector<shared_ptr<Net<Dtype>>> all_nets(){
			vector<shared_ptr<Net<Dtype>>> results;
			results.push_back(solver_->net());
			for (int i = 0; i < syncs.size(); i++)
			{
				results.push_back(syncs[i]->solver()->net());
			}
			return results;
		}
		inline shared_ptr<Net<Dtype>> test_net(){ return solver_->test_nets()[0]; }
		void ShareTrainedLayersWith(const Net<Dtype>* other);
	private:
		shared_ptr<Solver<Dtype>> solver_;
		std::vector<int> gpus;
		vector<shared_ptr<P2PSync<Dtype>>> syncs;
		shared_ptr<P2PSync<Dtype>> root_sync;
		static bool is_log_init_;
	};

	template<typename Dtype>
	shared_ptr<TrainManager<Dtype>> Train_Init_Load(string solver_proto, string gpu_ids="", string snapshot="", string weights="",string log_name="");
	template<typename Dtype>
	shared_ptr<TrainManager<Dtype>> Train_Init();
}
#endif