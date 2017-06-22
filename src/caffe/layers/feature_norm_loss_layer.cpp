#include <vector>

#include "caffe/layers/feature_norm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FeatureNormLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  cache_.ReshapeLike(*bottom[0]);
  vector<int> shape;
  shape.push_back(max(bottom[0]->channels(),bottom[0]->num()));
  ones_.Reshape(shape);
  caffe_set(ones_.count(), (Dtype)1.0, ones_.mutable_cpu_data());
  shape[0] = bottom[0]->num();
  cache2_.Reshape(shape);
}



#ifdef CPU_ONLY
STUB_GPU(FeatureNormLossLayer);
#endif

INSTANTIATE_CLASS(FeatureNormLossLayer);
REGISTER_LAYER_CLASS(FeatureNormLoss);

}  // namespace caffe
