#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_total_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void log_entropy_kernel(const int n, const Dtype* a, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = log(max(a[index], Dtype(FLT_MIN)));
	}
}


template <typename Dtype>
void EntropyTotalWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  Dtype* log_prob_data = prob_.mutable_gpu_diff();

  const Dtype* label = bottom[1]->gpu_data();
  const int inner_dim = prob_.count() / outer_num_;
  const int outer_dim = prob_.count() / inner_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  //log_entropy_kernel << <CAFFE_GET_BLOCKS(prob_.count()), CAFFE_CUDA_NUM_THREADS >> >(prob_.count(), prob_data, log_prob_data);
  //caffe_gpu_mul(prob_.count(), prob_data, log_prob_data, loss_data);
  Dtype* prob_history_data = prob_history_.mutable_gpu_data();
  Dtype* prob_cache = prob_history_.mutable_gpu_diff();

  //sum prob_data along num --> prob_cache
  if (inner_num_ > 1)
  {
	  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_dim, 1, inner_num_, (Dtype)1.0,
		  prob_data, inner_mul_.gpu_data(), (Dtype)0.0, cache_.mutable_gpu_data());
	  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, bottom[0]->channels(), outer_num_,
		  (Dtype)(1.0 - momemtum_), num_mul_.gpu_data(),
		  cache_.gpu_data(), (Dtype)0.0, prob_cache);
  }
  else
  {
	  //// prob_history expand to cache_
	  //caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, bottom[0]->channels(), 1,
		 // (Dtype)1.0, num_mul_.gpu_data(), prob_history_data,
		 // (Dtype)0.0, cache_.mutable_gpu_data());
	  //// add cache_ with prob_data
	  //caffe_gpu_add(cache_.count(), prob_data, cache_.gpu_data(), cache_.mutable_gpu_data());

	  //// prob_data divide cache_
	  //caffe_gpu_div(cache_.count(), prob_data, cache_.gpu_data(), cache_.mutable_gpu_diff());
	  //// sum cache_ along num to cache_ diff
	  //caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, bottom[0]->channels(), outer_num_,
		 // (Dtype)1.0, num_mul_.gpu_data(), cache_.gpu_data(), (Dtype)0.0, prob_cache);
	  //// add prob_cache diff with 0.5
	  //caffe_gpu_add_scalar(prob_history_.count(), (Dtype)0.5, prob_cache);
	  //// expand cache_ diff to cache_ diff
	  //caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, bottom[0]->channels(), 1,
		 // (Dtype)1.0, num_mul_.gpu_data(), prob_cache, (Dtype)0.0, cache_.mutable_gpu_diff());
	  //// cache_ divide cache_ diff.
	  //caffe_gpu_div(cache_.count(), cache_.gpu_data(), cache_.gpu_diff(), cache_.mutable_gpu_diff());
	  ////0.5 divide prob_cache
	  //caffe_gpu_div(prob_history_.count(), channel_mul_.gpu_data(), prob_cache, prob_cache);
	  //caffe_gpu_scal(prob_history_.count(), (Dtype)0.5, prob_cache);

	  //caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, bottom[0]->channels(), outer_num_,
		 // (Dtype)(1.0 - momemtum_) / outer_num_, num_mul_.gpu_data(),
		 // prob_data, (Dtype)0.0, prob_cache);
	  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, bottom[0]->channels(), outer_num_,
	   (Dtype)(1.0 - momemtum_), num_mul_.gpu_data(),
	   prob_data, (Dtype)0.0, prob_cache);
	  //caffe_gpu_mul(cache_.count(), prob_data, cache_.gpu_diff(), cache_.mutable_gpu_data());
	  //caffe_gpu_powx(prob_.count(), prob_data, (Dtype)1, cache_.mutable_gpu_diff());
	  ///*caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, bottom[0]->channels(), outer_num_,
		 // (Dtype)(1.0 - momemtum_), num_mul_.gpu_data(),
		 // cache_.gpu_data(), (Dtype)0.0, prob_cache); */
	  //caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, bottom[0]->channels(), outer_num_,
		 // (Dtype)(1.0 - momemtum_), num_mul_.gpu_data(),
		 // cache_.gpu_diff(), (Dtype)0.0, prob_cache);
  }

  //m*p(t-1)+(1-m)*prob_cache into prob_history_data
  caffe_gpu_scal(prob_history_.count(), (Dtype)momemtum_, prob_history_data);
  caffe_gpu_add(prob_history_.count(), prob_cache, prob_history_data, prob_history_data);
  // sum prob_history_data into norm_factor
  caffe_gpu_asum(prob_history_.count(), prob_history_data, &norm_factor_);
  // norm prob_history_data into prob_cache
  caffe_gpu_scale(prob_history_.count(), (Dtype)1.0 / norm_factor_, prob_history_data, prob_cache);
  //cal log_prob_data
  log_entropy_kernel << <CAFFE_GET_BLOCKS(prob_history_.count()), CAFFE_CUDA_NUM_THREADS >> >(prob_history_.count(), prob_cache, log_prob_data);

  // sum entropy loss
  Dtype loss;
  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, 1, prob_history_.count(), (Dtype)1.0,
	  prob_cache, log_prob_data, (Dtype)0.0, loss_data);
  loss = bottom[0]->cpu_diff()[0];

  top[0]->mutable_cpu_data()[0] = loss ;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void EntropyTotalWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* prob_data = prob_.mutable_gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    //caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
	Dtype* log_prob_data = prob_.mutable_gpu_diff();
    const Dtype* label = bottom[1]->gpu_data();
	const int inner_dim = prob_.count() / outer_num_;
	const int outer_dim = prob_.count() / inner_num_;
    const int nthreads = outer_num_ * inner_num_;

	//channel_sum_kernel << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
	//	nthreads, inner_num_, bottom[0]->channels(), bottom_diff);
	//caffe_gpu_sub(bottom[0]->count(), bottom_diff, log_prob_data, bottom_diff);
	//caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_data, bottom_diff);
	norm_factor_ = (1.0 - momemtum_) / norm_factor_;
	//(1-prob_data)*prob_data --> bottom_diff;
	caffe_gpu_scale(bottom[0]->count(), (Dtype)-1.0, prob_data, bottom_diff);
	caffe_gpu_add_scalar(bottom[0]->count(), (Dtype)1, bottom_diff);
	caffe_gpu_mul(bottom[0]->count(), prob_data, bottom_diff, bottom_diff);
	caffe_gpu_scal(bottom[0]->count(), (Dtype)norm_factor_, bottom_diff);

	// (log_prob_data-loss)
	Dtype* prob_history_data = prob_history_.mutable_gpu_data();
	caffe_gpu_add_scalar(prob_history_.count(), (Dtype)-top[0]->mutable_cpu_data()[0], log_prob_data);
	//expand log_prob_data.
	if (inner_num_ > 1)
	{
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, prob_history_.count(), 1, (Dtype)1.0,
			num_mul_.mutable_gpu_data(), log_prob_data, (Dtype)0.0, cache_.mutable_gpu_data());
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_dim, inner_num_, 1, (Dtype)1.0,
			cache_.gpu_data(), inner_mul_.gpu_data(), (Dtype)0.0, prob_data);
	}
	else
	{
		// scale prob
		//caffe_gpu_mul(bottom[0]->count(), cache_.gpu_diff(), bottom_diff, bottom_diff);
		//caffe_gpu_add_scalar(cache_.count(), (Dtype)-2, cache_.mutable_gpu_diff());
		//caffe_gpu_div(cache_.count(), cache_.gpu_diff(), prob_data, cache_.mutable_gpu_diff());
		//caffe_gpu_scal(cache_.count(), (Dtype)1.0, cache_.mutable_gpu_diff());
		//caffe_gpu_mul(bottom[0]->count(), cache_.gpu_diff(), bottom_diff, bottom_diff);

		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, prob_history_.count(), 1, (Dtype)1.0,
			num_mul_.mutable_gpu_data(), log_prob_data, (Dtype)0.0, prob_data);
	}
	caffe_gpu_mul(bottom[0]->count(), prob_data, bottom_diff, bottom_diff);


	Dtype loss_weight = top[0]->cpu_diff()[0];
	if (use_T_)
	{
		loss_weight /= (Dtype)temperature_;
	}
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyTotalWithLossLayer);

}  // namespace caffe
