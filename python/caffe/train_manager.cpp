#include "train_manager.h"
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

// Parse GPU ids or use all available devices
static void get_gpus(std::vector<int>* gpus, std::string FLAGS_gpu) {
	if (FLAGS_gpu == "all") {
		int count = 0;
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDeviceCount(&count));
#else
		NO_GPU;
#endif
		for (int i = 0; i < count; ++i) {
			gpus->push_back(i);
		}
	}
	else if (FLAGS_gpu.size()) {
		std::vector<std::string> strings;
		boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
		for (int i = 0; i < strings.size(); ++i) {
			gpus->push_back(boost::lexical_cast<int>(strings[i]));
		}
	}
	else {
		CHECK_EQ(gpus->size(), 0);
	}
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
	const std::string& flag_value) {
	if (flag_value == "stop") {
		return caffe::SolverAction::STOP;
	}
	if (flag_value == "snapshot") {
		return caffe::SolverAction::SNAPSHOT;
	}
	if (flag_value == "none") {
		return caffe::SolverAction::NONE;
	}
	LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
	return caffe::SolverAction::NONE;
}

template<typename Dtype>
// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<Dtype>* solver, const std::string& model_list) {
	std::vector<std::string> model_names;
	boost::split(model_names, model_list, boost::is_any_of(","));
	for (int i = 0; i < model_names.size(); ++i) {
		LOG(INFO) << "Finetuning from " << model_names[i];
		solver->net()->CopyTrainedLayersFrom(model_names[i]);
		for (int j = 0; j < solver->test_nets().size(); ++j) {
			solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
		}
	}
}

template void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list);
template void CopyLayers(caffe::Solver<double>* solver, const std::string& model_list);

namespace caffe{
	template<typename Dtype>
	shared_ptr<TrainManager<Dtype>> Train_Init(){
		shared_ptr<TrainManager<Dtype>> train(new TrainManager<Dtype>());
		return train;
	}

	template<typename Dtype>
	shared_ptr<TrainManager<Dtype>> Train_Init_Load(string solver_proto, string gpu_ids, string snapshot, string weights){
		shared_ptr<TrainManager<Dtype>> train(new TrainManager<Dtype>(solver_proto,gpu_ids,snapshot,weights));
		return train;
	}

	template shared_ptr<TrainManager<float>> Train_Init();
	template shared_ptr<TrainManager<double>> Train_Init();
	template shared_ptr<TrainManager<float>> Train_Init_Load(string solver_proto, string gpu_ids, string snapshot, string weights);
	template shared_ptr<TrainManager<double>> Train_Init_Load(string solver_proto, string gpu_ids, string snapshot, string weights);

	template<typename Dtype>
	shared_ptr<Net<Dtype>> TrainManager<Dtype>::Init(string solver_proto, string gpu_ids, string snapshot, string weights)
	{
		::google::InitGoogleLogging("TrainManager");

		CHECK_GT(solver_proto.size(), 0) << "Need a solver definition to train.";
		CHECK(!snapshot.size() || !weights.size())
			<< "Give a snapshot to resume training or weights to finetune "
			"but not both.";

		caffe::SolverParameter solver_param;
		caffe::ReadSolverParamsFromTextFileOrDie(solver_proto, &solver_param);

		// If the gpus flag is not provided, allow the mode and device to be set
		// in the solver prototxt.
		if (gpu_ids.size() == 0
			&& solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
			if (solver_param.has_device_id()) {
				gpu_ids = "" +
					boost::lexical_cast<string>(solver_param.device_id());
			}
			else {  // Set default GPU if unspecified
				gpu_ids = "" + boost::lexical_cast<string>(0);
			}
		}

		get_gpus(&gpus,gpu_ids);
		if (gpus.size() == 0) {
			LOG(INFO) << "Use CPU.";
			Caffe::set_mode(Caffe::CPU);
		}
		else {
			ostringstream s;
			for (int i = 0; i < gpus.size(); ++i) {
				s << (i ? ", " : "") << gpus[i];
			}
			LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
			cudaDeviceProp device_prop;
			for (int i = 0; i < gpus.size(); ++i) {
				cudaGetDeviceProperties(&device_prop, gpus[i]);
				LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
			}
#endif
			solver_param.set_device_id(gpus[0]);
			Caffe::SetDevice(gpus[0]);
			Caffe::set_mode(Caffe::GPU);
			Caffe::set_solver_count(gpus.size());
		}

		string sigint_effect = "stop";
		string sighup_effect = "snapshot";
		caffe::SignalHandler signal_handler(
			GetRequestedAction(sigint_effect),
			GetRequestedAction(sighup_effect));

		solver_.reset(caffe::SolverRegistry<Dtype>::CreateSolver(solver_param));

		solver_->SetActionFunction(signal_handler.GetActionFunction());

		if (snapshot.size()) {
			LOG(INFO) << "Resuming from " << snapshot;
			solver_->Restore(snapshot.c_str());
		}
		else if (weights.size()) {
			CopyLayers(solver_.get(), weights);
		}

		if (gpus.size() > 1) {
			root_sync.reset(new caffe::P2PSync<Dtype>(solver_, NULL, solver_->param()));
			syncs.resize(gpus.size());
			root_sync->Prepare(gpus, &syncs);
			//sync.Run(gpus);
			return root_sync->solver()->net();
		}
		else {
			//LOG(INFO) << "Starting Optimization";
			//solver->Solve();
			return solver_->net();
		}
	}

	template<typename Dtype>
	shared_ptr <Net<Dtype>> TrainManager<Dtype>::Train(int end_iter,shared_ptr<Net<Dtype>> net)
	{
		if (gpus.size() > 1)
		{
			for (int i = 1; i < syncs.size(); ++i) {
				syncs[i]->StartInternalThread_param(end_iter);
			}
			root_sync->solver()->Solve(NULL, end_iter);

			for (int i = 1; i < syncs.size(); ++i) {
				syncs[i]->StopInternalThread();
				syncs[i]->solver()->net()->Release_mem();
			}
			root_sync->solver()->net()->Release_mem();
			return root_sync->solver()->net();
		}
		else{
			solver_->Solve(NULL,end_iter);
			solver_->net()->Release_mem();
			return solver_->net();
		}
	}

	INSTANTIATE_CLASS(TrainManager);
}
