#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FilterEltLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void FilterEltLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0])
		return;
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(FilterEltLayer);

}  // namespace caffe
