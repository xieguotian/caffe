#include <vector>

#include "caffe/layers/feature_norm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureNormLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	caffe_gpu_mul(bottom[0]->count(),
		bottom[0]->gpu_data(),
		bottom[0]->gpu_data(),
		cache_.mutable_gpu_data());
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
		bottom[0]->num(), 1, bottom[0]->channels(), (Dtype)1.0,
		cache_.gpu_data(), ones_.gpu_data(),
		(Dtype)0.0, cache2_.mutable_gpu_data());

	caffe_gpu_sub(cache2_.count(), cache2_.gpu_data(), ones_.gpu_data(), cache2_.mutable_gpu_data());
	Dtype loss = cache2_.sumsq_data();
	top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / 4;
}

template <typename Dtype>
void FeatureNormLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num() / 4 ;
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
		bottom[0]->num(), bottom[0]->channels(), 1,
		(Dtype)alpha, cache2_.gpu_data(), ones_.gpu_data(),
		(Dtype)0.0, bottom[0]->mutable_gpu_diff());

	//caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_diff(), cache_.gpu_data(), bottom[0]->mutable_gpu_diff());
	caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());

}

INSTANTIATE_LAYER_GPU_FUNCS(FeatureNormLossLayer);

}  // namespace caffe
