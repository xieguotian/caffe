#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void mean_vector_clamp(int n_thread, Dtype* input_data,int out_ch, int in_ch, int h, int w)
	{
		CUDA_KERNEL_LOOP(index, n_thread) {
			int spat_idx = index % (w*h);
			int out_ch_idx = index / w / h;

			Dtype sum = 0;
			for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++)
			{
				int idx = (out_ch_idx*in_ch + in_ch_idx) *(w*h) + spat_idx;
				sum += input_data[idx];
			}
			sum = sum / in_ch;

			for (int in_ch_idx = 0; in_ch_idx < in_ch; in_ch_idx++)
			{
				int idx = (out_ch_idx*in_ch + in_ch_idx) *(w*h) + spat_idx;
				input_data[idx] -= sum;
				if (input_data[idx]>1)
					input_data[idx] = 1;
				if (input_data[idx] < -1)
					input_data[idx] = -1;
			}
		}
	}

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if (is_incremental_ && (!is_history_init_ ))//|| caffe::is_refresh_incremental))
	{
		int idx_param_idx = this->blobs_.size() - 1;
		caffe_copy(this->blobs_[0]->count(),
			this->blobs_[0]->gpu_data(),
			w_history_.mutable_gpu_data());
		is_history_init_ = true;
		//if (caffe::is_refresh_incremental)
		//{
		//	caffe_gpu_set(this->blobs_[idx_param_idx]->count(), (Dtype)0, this->blobs_[idx_param_idx]->mutable_gpu_data());
		//}
			//this->blobs_[idx_param_idx]->mutable_gpu_data());
	}
	if (is_incremental_)
	{
		int idx_param_idx = this->blobs_.size() - 1;
		caffe_gpu_add(this->blobs_[0]->count(),
			w_history_.gpu_data(),
			this->blobs_[idx_param_idx]->gpu_data(),
			this->blobs_[0]->mutable_gpu_data());
	}

  const Dtype* weight = this->blobs_[0]->gpu_data();

  if (this->layer_param_.convolution_param().is_binarized_param() && !this->blobs_[0]->is_binarized()) 
  {
	  static bool is_first = true;
	  if (is_first)
	  {
		  LOG(INFO) << "use binary weight for training";
		  is_first = false;
	  }
	  int n_thread = this->blobs_[0]->num()*this->blobs_[0]->height()*this->blobs_[0]->width();

	  if (this->phase_ == TRAIN)
	  {
		  // sub mean and clamp to [-1,1]
		  mean_vector_clamp<Dtype> << <CAFFE_GET_BLOCKS(n_thread), CAFFE_CUDA_NUM_THREADS >> > (
			  n_thread, this->blobs_[0]->mutable_gpu_data(), this->blobs_[0]->num(),
			  this->blobs_[0]->channels(), this->blobs_[0]->height(), this->blobs_[0]->width()
			  );
	  }
	  // binarize weight
	  caffe_gpu_sign(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), sign_weight_.mutable_gpu_data());
	  // calculate abs(weight)
	  caffe_gpu_abs<Dtype>(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), sign_weight_.mutable_gpu_diff());
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
		  this->blobs_[0]->num(), 1,
		  this->blobs_[0]->channels()*this->blobs_[0]->height()*this->blobs_[0]->width(),
		  (Dtype)1.0 / this->blobs_[0]->num(),
		  //this->blobs_[0]->gpu_data(),
		  sign_weight_.gpu_diff(),
		  sum_cache_.gpu_data(),
		  (Dtype)0.0,
		  sum_result_.mutable_gpu_data()
		  );
	  
	  //approximate real weight.
	  num_mul_kernel<Dtype> << <CAFFE_GET_BLOCKS(this->blobs_[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
		  this->blobs_[0]->count(), this->blobs_[0]->channels(), this->blobs_[0]->height()* this->blobs_[0]->width(),
		  sign_weight_.gpu_data(), sum_result_.gpu_data(), sign_weight_.mutable_gpu_data()
		  );
	  weight = sign_weight_.gpu_data();
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }

  if (is_direct_connect_)
  {
	  int idx_param_idx = this->blobs_.size() - 1;
	  for (int i = 0; i < bottom.size(); ++i)
	  {
		  const Dtype* bottom_data = bottom[i]->gpu_data();
		  Dtype* top_data = top[i]->mutable_gpu_data();
		  for (int idx = 0; idx < this->blobs_[idx_param_idx]->count(); ++idx)
		  {
			  int sel_idx = this->blobs_[idx_param_idx]->cpu_data()[idx];
			  for (int n = 0; n < bottom[i]->num(); ++n)
			  {
				  caffe_copy(bottom[i]->count(2),
					  bottom_data + bottom[i]->offset(n, sel_idx),
					  top_data + top[i]->offset(n, idx + num_output_));
			  }
		  }
	  }
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* weight = NULL;
	Dtype* weight_diff = NULL;
	if (this->param_propagate_down_[0]) {
		weight = this->blobs_[0]->gpu_data();
		weight_diff = this->blobs_[0]->mutable_gpu_diff();
	}
	Dtype* bias_diff = NULL;
	if (this->bias_term_ && this->param_propagate_down_[1]) {
		bias_diff = this->blobs_[1]->mutable_gpu_diff();
	}
	for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = top[i]->gpu_diff();
		const Dtype* top_data = top[i]->gpu_data();
		// Backward through cuDNN in parallel over groups and gradients.
		for (int g = 0; g < this->group_; g++) {
			// Gradient w.r.t. bias.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0 * this->group_ + g],
					cudnn::dataType<Dtype>::one,
					top_descs_[i], top_diff + top_offset_ * g,
					cudnn::dataType<Dtype>::one,
					bias_desc_, bias_diff + bias_offset_ * g));
			}

			// Gradient w.r.t. weights.
			if (this->param_propagate_down_[0]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				CUDNN_CHECK(cudnnConvolutionBackwardFilter(
					handle_[1 * this->group_ + g],
					cudnn::dataType<Dtype>::one,
					bottom_descs_[i], bottom_data + bottom_offset_ * g,
					top_descs_[i], top_diff + top_offset_ * g,
					conv_descs_[i],
					bwd_filter_algo_[i], workspace[1 * this->group_ + g],
					workspace_bwd_filter_sizes_[i],
					cudnn::dataType<Dtype>::one,
					filter_desc_, weight_diff + this->weight_offset_ * g));
			}

			// Gradient w.r.t. bottom data.
			if (propagate_down[i]) {
				if (weight == NULL) {
					weight = this->blobs_[0]->gpu_data();
				}
				if (this->layer_param_.convolution_param().is_binarized_param())
				{
					static bool is_first = true;
					if (is_first)
					{
						LOG(INFO) << "use binary weight for backward";
						is_first = false;
					}
					weight = sign_weight_.gpu_data();
				}
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				CUDNN_CHECK(cudnnConvolutionBackwardData(
					handle_[2 * this->group_ + g],
					cudnn::dataType<Dtype>::one,
					filter_desc_, weight + this->weight_offset_ * g,
					top_descs_[i], top_diff + top_offset_ * g,
					conv_descs_[i],
					bwd_data_algo_[i], workspace[2 * this->group_ + g],
					workspace_bwd_data_sizes_[i],
					cudnn::dataType<Dtype>::zero,
					bottom_descs_[i], bottom_diff + bottom_offset_ * g));
			}
		}

		// Synchronize the work across groups, each of which went into its own
		// stream, by launching an empty kernel into the default (null) stream.
		// NOLINT_NEXT_LINE(whitespace/operators)
		sync_conv_groups << <1, 1 >> >();
	}

	Dtype decay_mult = this->layer_param_.decay_mult();
	if (decay_mult > 0)
	{
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			const Dtype* top_data = top[i]->gpu_data();
			// Backward through cuDNN in parallel over groups and gradients.
			for (int g = 0; g < this->group_; g++) {
				// Gradient w.r.t. weights.
				if (this->param_propagate_down_[0]) {
					const Dtype* bottom_data = bottom[i]->gpu_data();
					// backpropagate  signal decay.
					CUDNN_CHECK(cudnnConvolutionBackwardFilter(
						handle_[1 * this->group_ + g],
						//cudnn::dataType<Dtype>::one,
						&decay_mult,
						bottom_descs_[i], bottom_data + bottom_offset_ * g,
						//top_descs_[i], top_diff + top_offset_ * g,
						top_descs_[i], top_data + top_offset_*g,
						conv_descs_[i],
						bwd_filter_algo_[i], workspace[1 * this->group_ + g],
						workspace_bwd_filter_sizes_[i],
						cudnn::dataType<Dtype>::one,
						filter_desc_, weight_diff + this->weight_offset_ * g));
				}
			}
			sync_conv_groups << <1, 1 >> >();
		}
	}

  if (is_direct_connect_)
  {
	  int idx_param_idx = this->blobs_.size() - 1;
	  for (int i = 0; i < bottom.size(); ++i)
	  {
		  Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		  const Dtype* top_diff = top[i]->gpu_diff();
		  for (int idx = 0; idx < this->blobs_[idx_param_idx]->count(); ++idx)
		  {
			  int sel_idx = this->blobs_[idx_param_idx]->cpu_data()[idx];
			  for (int n = 0; n < bottom[i]->num(); ++n)
			  {
				  caffe_gpu_axpy(bottom[i]->count(2), (Dtype)1.0,
					  top_diff + top[i]->offset(n, idx + num_output_),
					  bottom_diff + bottom[i]->offset(n, sel_idx));
			  }
		  }
	  }
  }
  if (is_incremental_)
  {
	  int idx_param_idx = this->blobs_.size() - 1;
	  caffe_copy(this->blobs_[0]->count(),
		  this->blobs_[0]->gpu_diff(),
		  this->blobs_[idx_param_idx]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
