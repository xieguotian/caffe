#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void cross_entropy_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = -b[index] * log(max(a[index], Dtype(FLT_MIN)));
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
  cross_entropy_kernel<Dtype> << <bottom[0]->count(), CAFFE_CUDA_NUM_THREADS >> >(
	  bottom[0]->count(),
	  prob_data, label, 
	  loss_data);

  Dtype loss;
  caffe_gpu_asum(bottom[0]->count(), loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
	  outer_num_*inner_num_);
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
	if (use_T_)
	{
		caffe_gpu_scal(bottom[0]->count(), (Dtype)1.0 / temperature_, bottom_diff);
	}

    const Dtype loss_weight = top[0]->cpu_diff()[0] /
		get_normalizer(normalization_, outer_num_*inner_num_);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);

}  // namespace caffe
