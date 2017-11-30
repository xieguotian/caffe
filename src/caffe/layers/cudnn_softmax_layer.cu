#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_data;
	if (this->use_T_)
	{
		bottom_data = this->cache_.mutable_gpu_data();//bottom[0]->mutable_gpu_diff();
		caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), bottom_data);
		caffe_gpu_scal(bottom[0]->count(), (Dtype)1.0 / this->temperature_, bottom_data);
	}
	else
	{
		bottom_data = bottom[0]->mutable_gpu_data();
	}
  Dtype* top_data = top[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnSoftmaxForward(handle_, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        cudnn::dataType<Dtype>::one,
        bottom_desc_, bottom_data,
        cudnn::dataType<Dtype>::zero,
        top_desc_, top_data));
  //if (use_T_)
	 // caffe_gpu_set(bottom[0]->count(), (Dtype)0.0, bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    CUDNN_CHECK(cudnnSoftmaxBackward(handle_, CUDNN_SOFTMAX_ACCURATE,
          CUDNN_SOFTMAX_MODE_CHANNEL,
          cudnn::dataType<Dtype>::one,
          top_desc_, top_data, top_desc_, top_diff,
          cudnn::dataType<Dtype>::zero,
          bottom_desc_, bottom_diff));
	if (this->use_T_)
		caffe_gpu_scal(top[0]->count(), (Dtype)1.0 / this->temperature_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNSoftmaxLayer);

}  // namespace caffe
#endif
