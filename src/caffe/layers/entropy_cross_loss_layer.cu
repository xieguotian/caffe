#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_cross_loss_layer.hpp"
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
__global__ void log_entropy_kernel(const int n, const Dtype* a, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = log(max(a[index], Dtype(FLT_MIN)));
	}
}

template <typename Dtype>
void EntropyCrossWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  Dtype* log_prob_data = prob_.mutable_gpu_diff();
  Dtype* cross_prob_data = cross_prob_.mutable_gpu_data();

  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  log_entropy_kernel << <CAFFE_GET_BLOCKS(prob_.count()), CAFFE_CUDA_NUM_THREADS >> >(prob_.count(), prob_data, log_prob_data);
  caffe_gpu_mul(prob_.count(), prob_data, log_prob_data, loss_data);
  caffe_gpu_gemm(CblasNoTrans,
	  CblasTrans,
	  outer_num_,
	  outer_num_,
	  bottom[0]->channels(),
	  (Dtype)1,
	  prob_data,
	  log_prob_data,
	  (Dtype)0,
	  cross_prob_data);

  Dtype loss,loss2;
  caffe_gpu_asum(cross_prob_.count(), cross_prob_data, &loss);
  caffe_gpu_asum(bottom[0]->count(), loss_data, &loss2);
  loss = -(loss - loss2);

  Dtype valid_count = (outer_num_ - 1)*outer_num_;//outer_num_*inner_num_;

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
void EntropyCrossWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
	Dtype* num_sum_data = cache_.mutable_gpu_data();
	Dtype* num_sum_diff = cache_.mutable_gpu_diff();

    //caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
	const Dtype* log_prob_data = prob_.gpu_diff();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;

	//channel_sum_kernel << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
	//	nthreads, inner_num_, bottom[0]->channels(), bottom_diff);
	//caffe_gpu_sub(bottom[0]->count(), bottom_diff, log_prob_data, bottom_diff);
	//caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_data, bottom_diff);

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1,
		bottom[0]->channels(), outer_num_,
		(Dtype)1, num_mul_.gpu_data(), log_prob_data,
		(Dtype)1, num_sum_data);
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_,
		bottom[0]->channels(),1 ,
		(Dtype)1, num_mul_.gpu_data(), num_sum_data,
		(Dtype)1, num_sum_diff);

	caffe_gpu_sub(bottom[0]->count(), log_prob_data, num_sum_diff, num_sum_diff);
	caffe_gpu_mul(bottom[0]->count(), prob_data, num_sum_diff, num_sum_data);

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_,
		1,bottom[0]->channels(),
		(Dtype)1, num_sum_data, num_mul_.gpu_data(),
		(Dtype)0, cache2_.mutable_gpu_data());
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_,
		bottom[0]->channels(), 1,
		(Dtype)1, cache2_.gpu_data(),channel_mul_.gpu_data(),
		(Dtype)0, num_sum_data);
	caffe_gpu_sub(bottom[0]->count(), num_sum_data, num_sum_diff, bottom_diff);
	caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_data, bottom_diff);
	caffe_gpu_axpy(bottom[0]->count(), (Dtype)-outer_num_, prob_data, bottom_diff);
	
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1,
		bottom[0]->channels(), outer_num_,
		(Dtype)1, num_mul_.gpu_data(), prob_data,
		(Dtype)0, cache2_.mutable_gpu_data());
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, outer_num_,
		bottom[0]->channels(), 1,
		(Dtype)1, num_mul_.gpu_data(), cache2_.mutable_gpu_data(),
		(Dtype)0, cache2_.mutable_gpu_diff());
	caffe_gpu_add(bottom[0]->count(), bottom_diff, cache2_.mutable_gpu_diff(), bottom_diff);

	const Dtype loss_weight = top[0]->cpu_diff()[0] / ((outer_num_ - 1)*outer_num_);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyCrossWithLossLayer);

}  // namespace caffe
