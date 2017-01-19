#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kmeans_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KmeansLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter cluster_param(this->layer_param_);
  cluster_param.set_type("ClusterCentroidDist");
  //cluster_centroid_dist_layer = LayerRegistry<Dtype>::CreateLayer(cluster_param);
  //distance_bottom_vec_.clear();
  //distance_bottom_vec_.push_back(bottom[0]);
  //distance_top_vec_.clear();
  //distance_top_vec_.push_back(&distance_);
  //cluster_centroid_dist_layer->SetUp(distance_bottom_vec_, distance_top_vec_);

}

template <typename Dtype>
void KmeansLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //cluster_centroid_dist_layer->Reshape(distance_bottom_vec_, distance_top_vec_);

  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  vector<int> shape;
  shape.push_back(bottom[0]->num());
  pos_.Reshape(shape);
  max_value_set_.Reshape(shape);
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(KmeansLossLayer);
REGISTER_LAYER_CLASS(KmeansLoss);

}  // namespace caffe
