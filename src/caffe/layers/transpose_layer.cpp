#include <vector>

#include "caffe/layers/transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> top_shape(2);
	top_shape[0] = bottom[0]->num()*bottom[0]->height()*bottom[0]->width();
	top_shape[1] = bottom[0]->channels();
	top[0]->Reshape(top_shape);

	size_x_ = bottom[0]->num()*bottom[0]->channels();
	size_y_ = bottom[0]->height()*bottom[0]->width();

}

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe
