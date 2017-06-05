#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void log_entropy_kernel(const int n, const Dtype* a, Dtype* y,const int channels,const Dtype* label=NULL,const Dtype* pred_label=NULL) {
	CUDA_KERNEL_LOOP(index, n) {
		if (label != NULL && pred_label != NULL)
		{
			int idx = index / channels;
			if (label[idx] == pred_label[idx])
			{
				y[index] = log(max(a[index], Dtype(FLT_MIN)));
			}
			else
				y[index] = 0;
		}
		else{
			y[index] = log(max(a[index], Dtype(FLT_MIN)));
		}
	}
}

template <typename Dtype>
void EntropyWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  Dtype* log_prob_data = prob_.mutable_gpu_diff();

  bool has_label = bottom.size() >= 3;
  const Dtype* label = has_label ? bottom[1]->gpu_data() : NULL;
  const Dtype* pred_label = has_label ? bottom[2]->gpu_data() : NULL;
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  log_entropy_kernel << <CAFFE_GET_BLOCKS(prob_.count()), CAFFE_CUDA_NUM_THREADS >> >(prob_.count(), 
	  prob_data, 
	  log_prob_data,
	  prob_.channels(),
	  label,
	  pred_label);
  caffe_gpu_mul(prob_.count(), prob_data, log_prob_data, loss_data);

  Dtype loss;
  caffe_gpu_asum(bottom[0]->count(), loss_data, &loss);
  Dtype valid_count = outer_num_*inner_num_;

  int num_count = 0;
  Dtype norm_weight = nthreads;
  if (has_label)
  {
	  const Dtype* label = has_label ? bottom[1]->cpu_data() : NULL;
	  const Dtype* pred_label = has_label ? bottom[2]->cpu_data() : NULL;
	  for (int i = 0; i < bottom[1]->count(); i++)
	  {
		  if (label[i] == pred_label[i])
			  num_count += 1;
	  }
	  valid_count = num_count+1e-10;
  }

  top[0]->mutable_cpu_data()[0] = loss / valid_count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
__global__ void channel_sum_kernel(const int n, const int spat_dim, const int channels, Dtype* a) {
	CUDA_KERNEL_LOOP(index, n) {
		int n_idx = index / spat_dim;
		int spat_idx = index % spat_dim;

		Dtype sum = 0;
		for (int ch = 0; ch < channels; ++ch)
		{
			int idx = (n_idx*channels + ch)*spat_dim + spat_idx;
			sum += a[idx];
		}
		
		for (int ch = 0; ch < channels; ++ch)
		{
			int idx = (n_idx*channels + ch)*spat_dim + spat_idx;
			a[idx] = sum;
		}

	}
}

template <typename Dtype>
void EntropyWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    //caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
	const Dtype* log_prob_data = prob_.gpu_diff();
	bool has_label = bottom.size() >= 3;
	const Dtype* label = has_label ? bottom[1]->cpu_data() : NULL;
	const Dtype* pred_label = has_label ? bottom[2]->cpu_data() : NULL;
    const int outer_dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;

	int num_count = 0;
	Dtype norm_weight = nthreads;
	if (has_label)
	{
		for (int i = 0; i < bottom[1]->count(); i++)
		{
			if (label[i] == pred_label[i])
				num_count += 1;
		}
		norm_weight = num_count + 1e-10;
	}
	//channel_sum_kernel << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
	//	nthreads, inner_num_, bottom[0]->channels(), bottom_diff);
	//caffe_gpu_sub(bottom[0]->count(), bottom_diff, log_prob_data, bottom_diff);
	//caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_data, bottom_diff);
	if (inner_num_ > 1)
	{
		for (int n = 0; n < outer_num_; ++n)
		{
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, inner_num_, bottom[0]->channels(),
				(Dtype)1.0, channel_mul_.gpu_data(), bottom_diff + n*outer_dim, (Dtype)0,
				cache_.mutable_gpu_data()+n*inner_num_);
		}
		for (int n = 0; n < outer_num_; ++n)
		{
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), inner_num_, 1,
				(Dtype)1.0, channel_mul_.gpu_data(), cache_.gpu_data() + n*inner_num_,
				(Dtype)0, cache_.mutable_gpu_diff() + n*outer_dim);
		}
	}
	else
	{
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, 1, bottom[0]->channels(),
			(Dtype)1.0, bottom_diff, channel_mul_.gpu_data(), (Dtype)0, cache_.mutable_gpu_data());
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_, bottom[0]->channels(), 1,
			(Dtype)1.0, cache_.gpu_data(), channel_mul_.gpu_data(), (Dtype)0, cache_.mutable_gpu_diff());
	}
	caffe_gpu_sub(bottom[0]->count(), cache_.gpu_diff(), log_prob_data, bottom_diff);
	caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_data, bottom_diff);
	Dtype loss_weight = top[0]->cpu_diff()[0] / norm_weight; // / nthreads;
	if (use_T_)
	{
		loss_weight /= (Dtype)temperature_;
	}
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
	LOG(INFO) << bottom[0]->asum_diff();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyWithLossLayer);

}  // namespace caffe
