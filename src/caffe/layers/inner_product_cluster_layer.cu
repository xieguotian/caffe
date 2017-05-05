#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_cluster_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductClusterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
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

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* labels = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* weight_tmp;
  Dtype* top_cache_data = top_cache_.mutable_gpu_data();
  for (int n = 0; n < M_; n++)
  {
	  int label = labels[n];
	  weight_tmp = weight + label*N_*K_;
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
		  weight_tmp, bottom_data + bottom[0]->offset(n), (Dtype)0., 
		  top_data + top[0]->offset(n));
	  caffe_gpu_gemv<Dtype>(CblasTrans, N_, K_, (Dtype)1.,
		  weight_tmp, top_data + top[0]->offset(n), (Dtype)0.,
		  top_cache_data + top_cache_.offset(n));
  }
  ///LOG(INFO) << "before:"<< top_cache_.cpu_diff()[0] << " " << top_cache_.cpu_data()[0];
  caffe_gpu_sub(bottom[0]->count(), top_cache_data, bottom_data, top_cache_.mutable_gpu_diff());
  ///LOG(INFO) << top_cache_.cpu_diff()[0];
  Dtype loss = top_cache_.sumsq_diff();
  if (top.size() > 1)
  {
	  //LOG(INFO) << "loss: " << loss << "  wegith: "<< weight_data;
	  top[1]->mutable_cpu_data()[0] = loss;
  }
  if (M_ == 1) {
    //caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
    //                     weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    //caffe_gpu_gemm<Dtype>(CblasNoTrans,
    //                      transpose_ ? CblasNoTrans : CblasTrans,
    //                      M_, N_, K_, (Dtype)1.,
    //                      bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductClusterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* labels = bottom[1]->cpu_data();
	const Dtype* top_cache_diff = top_cache_.gpu_diff();
	Dtype* top_cache2_data = top_cache2_.mutable_gpu_data();
	const Dtype* top_data = top[0]->gpu_data();
    // Gradient with respect to weight
	const Dtype* weight_tmp;
	Dtype lr = 1;
	if (layer_param_.loss_weight_size() > 0)
		lr = layer_param_.loss_weight(0);
	for (int n = 0; n < M_; n++)
	{
		int label = labels[n];
		weight_tmp = this->blobs_[0]->gpu_data() + label*N_*K_;
		if (transpose_) {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				K_, N_, 1,
				(Dtype)1., bottom_data + bottom[0]->offset(n), top_diff + top[0]->offset(n),
				(Dtype)1., this->blobs_[0]->mutable_gpu_diff() + label*M_*K_);
		}
		else {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				N_, K_, 1,
				(Dtype)1., top_diff + top[0]->offset(n), bottom_data + bottom[0]->offset(n),
				(Dtype)1., this->blobs_[0]->mutable_gpu_diff() + label*M_*K_);

			// top_data: n*N_ 
			// top_cache_diff: n*K_
			//LOG(INFO) << (top[0]->cpu_data() + top[0]->offset(n))[0] << " " << (top_cache_.cpu_diff()+top_cache_.offset(n))[0];
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				N_, K_, 1,
				(Dtype)lr , top_data + top[0]->offset(n), top_cache_diff + top_cache_.offset(n),
				(Dtype)1., this->blobs_[0]->mutable_gpu_diff() + label*M_*K_);

			caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
				weight_tmp, top_cache_diff + top_cache_.offset(n), (Dtype)0.,
				top_cache2_data + top_cache2_.offset(n));
			// top_cache2_data: n*N_
			// bottom_data: n*K_;
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				N_, K_, 1,
				(Dtype)lr , top_cache2_data + top_cache2_.offset(n), bottom_data + bottom[0]->offset(n),
				(Dtype)1., this->blobs_[0]->mutable_gpu_diff() + label*M_*K_);
		}
	}
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
	const Dtype* labels = bottom[1]->cpu_data();
	for (int n = 0; n < M_; n++)
	{
		int label = labels[n];
		const Dtype* weight_tmp = this->blobs_[0]->gpu_data() + label*N_*K_;
		if (transpose_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M_, K_, N_,
				(Dtype)1., top_diff+top[0]->offset(n), weight_tmp,//#this->blobs_[0]->gpu_data(),
				(Dtype)0., bottom[0]->mutable_gpu_diff()+bottom[0]->offset(n));
		}
		else {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
				M_, K_, N_,
				(Dtype)1., top_diff+top[0]->offset(n), weight_tmp,//this->blobs_[0]->gpu_data(),
				(Dtype)0., bottom[0]->mutable_gpu_diff()+bottom[0]->offset(n));
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

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductClusterLayer);

}  // namespace caffe
