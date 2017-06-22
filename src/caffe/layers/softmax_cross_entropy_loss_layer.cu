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
  softmax_layer_->Forward(softmax_bottom_vec_label_, softmax_top_vec_label_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  label = label_.gpu_data();
  
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  Dtype* counts = prob_.mutable_gpu_diff();
  const Dtype* hard_label = NULL;
  if (bottom.size() == 3)
	  hard_label = bottom[2]->gpu_data();

  cross_entropy_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
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
  if (is_update_T_ && top.size() == 2)
  {
	  top[1]->mutable_cpu_data()[0] = this->blobs_[0]->cpu_data()[0];
  }
  else
  {
	  if (top.size() == 2) {
		  top[1]->ShareData(prob_);
	  }
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
	label = label_.gpu_data();

	caffe_gpu_sub(bottom[0]->count(), prob_data, label, bottom_diff);
	if (has_ignore_label_ && bottom.size() == 3)
		caffe_gpu_mul(bottom[0]->count(), bottom_diff, prob_.gpu_diff(), bottom_diff);
	//if (use_T_)
	//{
	//	caffe_gpu_scal(bottom[0]->count(), (Dtype)1.0 / temperature_, bottom_diff);
	//}

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

	Dtype diff_sum;
	Dtype prob_org_sum;
	static bool first_time_norm = true;
	softmax_layer_org_->Forward(softmax_bottom_vec_org_, softmax_top_vec_org_);
	if (bottom.size() == 3 && !has_ignore_label_ && gradient_norm_ == SoftmaxParameter_GradientNorm_HARD_NORM)
	{
		if (first_time_norm)
		{
			LOG(INFO) << "cross entropy loss with HARD_NROM";
			first_time_norm = false;
		}
		caffe_set(label_.count(), (Dtype)0.0, label_.mutable_cpu_data());
		Dtype* label_ptr = label_.mutable_cpu_data();
		int stride = label_.channels();
		for (int idx = 0; idx < bottom[2]->count(); ++idx)
		{
			int tmp_label = bottom[2]->cpu_data()[idx];
			label_ptr[idx*stride + tmp_label] = 1;
		}
		label = label_.gpu_data();
		caffe_gpu_sub(prob_org_.count(), prob_org_.gpu_data(), label, prob_org_.mutable_gpu_data());
		for (int idx = 0; idx < bottom[0]->num(); idx++)
		{
			caffe_gpu_asum(bottom[0]->channels(), prob_org_.gpu_data() + idx*bottom[0]->channels(), &prob_org_sum);
			caffe_gpu_asum(bottom[0]->channels(), bottom_diff + idx*bottom[0]->channels(), &diff_sum);
			Dtype scale_factor = prob_org_sum / diff_sum;
			caffe_gpu_scal(bottom[0]->channels(), scale_factor, bottom_diff + idx*bottom[0]->channels());
		}
		Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
	}
	else if (bottom.size() == 3 && !has_ignore_label_ && gradient_norm_ == SoftmaxParameter_GradientNorm_EASY_NORM)
	{
		if (first_time_norm)
		{
			LOG(INFO) << "cross entropy loss with EASY_NROM";
			first_time_norm = false;
		}
		caffe_set(label_.count(), (Dtype)0.0, label_.mutable_cpu_data());
		Dtype* label_ptr = label_.mutable_cpu_data();
		int stride = label_.channels();
		for (int idx = 0; idx < bottom[2]->count(); ++idx)
		{
			int tmp_label = bottom[2]->cpu_data()[idx];
			label_ptr[idx*stride + tmp_label] = 1;
		}
		label = label_.gpu_data();
		caffe_gpu_sub(prob_org_.count(), prob_org_.gpu_data(), label, prob_org_.mutable_gpu_data());

		caffe_gpu_asum(bottom[0]->count(), prob_org_.gpu_data(), &prob_org_sum);
		caffe_gpu_asum(bottom[0]->count(), bottom_diff, &diff_sum);
		Dtype scale_factor = prob_org_sum / diff_sum;

		Dtype loss_weight = scale_factor * top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
	}
	else if (bottom.size() == 3 && !has_ignore_label_ && gradient_norm_ == SoftmaxParameter_GradientNorm_EQUAL_NORM)
	{
		if (first_time_norm)
		{
			LOG(INFO) << "cross entropy loss with EQUAL_NROM";
			first_time_norm = false;
		}
		caffe_set(label_.count(), (Dtype)0.0, label_.mutable_cpu_data());
		Dtype* label_ptr = label_.mutable_cpu_data();
		int stride = label_.channels();
		for (int idx = 0; idx < bottom[2]->count(); ++idx)
		{
			int tmp_label = bottom[2]->cpu_data()[idx];
			label_ptr[idx*stride + tmp_label] = 1;
		}
		label = label_.gpu_data();
		caffe_gpu_sub(prob_org_.count(), prob_org_.gpu_data(), label, prob_org_.mutable_gpu_data());
		caffe_gpu_asum(bottom[0]->count(), prob_org_.gpu_data(), &prob_org_sum);
		prob_org_sum = prob_org_sum / bottom[0]->num();
		for (int idx = 0; idx < bottom[0]->num(); idx++)
		{
			caffe_gpu_asum(bottom[0]->channels(), bottom_diff + idx*bottom[0]->channels(), &diff_sum);
			Dtype scale_factor = prob_org_sum / diff_sum;
			caffe_gpu_scal(bottom[0]->channels(), scale_factor, bottom_diff + idx*bottom[0]->channels());
		}
		Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
	}
	else{
		if (first_time_norm)
		{
			LOG(INFO) << "cross entropy loss with DEFAULT_NROM";
			first_time_norm = false;
		}
		softmax_layer_org_->Forward(softmax_bottom_vec_label_, softmax_top_vec_label_);
		label = label_.gpu_data();
		caffe_gpu_sub(prob_org_.count(), prob_org_.gpu_data(), label, prob_org_.mutable_gpu_data());
		caffe_gpu_asum(bottom[0]->count(), bottom_diff, &diff_sum);
		caffe_gpu_asum(bottom[0]->count(), prob_org_.gpu_data(), &prob_org_sum);
		Dtype scale_factor = prob_org_sum / diff_sum;

		const Dtype loss_weight = scale_factor * top[0]->cpu_diff()[0] /
			get_normalizer(normalization_, valid_count);
		caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
	}

	if (is_update_T_ && this->blobs_[0]->cpu_data()[0]>1)
	{
		//LOG(INFO) << this->blobs_[0]->mutable_cpu_data()[0] << "," << update_step_;
		this->blobs_[0]->mutable_cpu_data()[0] = this->blobs_[0]->cpu_data()[0] - update_step_;
	}
	else if (is_update_T_ && this->blobs_[0]->cpu_data()[0] < 1)
	{
		this->blobs_[0]->mutable_cpu_data()[0] = 1;
	}
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);

}  // namespace caffe
