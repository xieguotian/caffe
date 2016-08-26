#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void cross_entropy_kernel(const int n, const Dtype* a, 
	const Dtype* b, Dtype* y, Dtype* counts,
	const bool has_ignore_label_, const int ignore_label_, 
	const int spatial_dim, const int channels,
	const Dtype* label) {
	CUDA_KERNEL_LOOP(index, n) {
		const int n = index / spatial_dim / channels;
		const int s = index % spatial_dim;
		int label_value;
		if (label!=NULL)
			label_value = static_cast<int>(label[n * spatial_dim + s]);

		if (label!=NULL && has_ignore_label_ && label_value != ignore_label_) {
			y[index] = 0;
			counts[index] = 0;
		}
		else{
			y[index] = -b[index] * log(max(a[index], Dtype(FLT_MIN)));
			counts[index] = 1;
		}
	}
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  Dtype* counts = prob_.mutable_gpu_diff();
  const Dtype* hard_label = NULL;
  if (bottom.size() == 3)
	  hard_label = bottom[2]->gpu_data();

  cross_entropy_kernel<Dtype> << <bottom[0]->count(), CAFFE_CUDA_NUM_THREADS >> >(
	  bottom[0]->count(),
	  prob_data, label, 
	  loss_data, counts, 
	  has_ignore_label_, ignore_label_,
	  inner_num_, bottom[0]->channels(),
	  hard_label);

  Dtype loss;
  caffe_gpu_asum(bottom[0]->count(), loss_data, &loss);

  Dtype valid_count = outer_num_*inner_num_;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
	  has_ignore_label_ && bottom.size() == 3) {
	  caffe_gpu_asum(prob_.count(), counts, &valid_count);
	  valid_count /= bottom[0]->channels();
  }

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
	  valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}


template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();

	caffe_gpu_sub(bottom[0]->count(), prob_data, label, bottom_diff);
	if (has_ignore_label_ && bottom.size() == 3)
		caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_.gpu_diff(), bottom_diff);
	if (use_T_)
	{
		caffe_gpu_scal(bottom[0]->count(), (Dtype)1.0 / temperature_, bottom_diff);
	}

	// Since this memory is never used for anything else,
	// we use to to avoid allocating new GPU memory.
	Dtype* counts = prob_.mutable_gpu_diff();

	Dtype valid_count = outer_num_*inner_num_;
	// Only launch another CUDA kernel if we actually need the count of valid
	// outputs.
	if (normalization_ == LossParameter_NormalizationMode_VALID &&
		has_ignore_label_ && bottom.size() == 3) {
		caffe_gpu_asum(prob_.count(), counts, &valid_count);
		valid_count /= bottom[0]->channels();
	}

    const Dtype loss_weight = top[0]->cpu_diff()[0] /
		get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);

}  // namespace caffe
