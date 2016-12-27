#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cross_entropy_m_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxCrossEntropyMLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  use_T_ = this->layer_param_.softmax_param().temperature() != 0;
  temperature_ = this->layer_param_.softmax_param().temperature();
}

template <typename Dtype>
void SoftmaxCrossEntropyMLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxCrossEntropyMLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxCrossEntropyMLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;

  for (int idx = 0; idx < bottom[0]->count(); ++idx)
  {
	  loss -= label[idx] * log(max(prob_data[idx], Dtype(FLT_MIN)));
  }

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, outer_num_*inner_num_);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    
	caffe_sub(bottom[0]->count(), prob_data, label, bottom_diff);
	if (use_T_)
	{
		caffe_scal(bottom[0]->count(), (Dtype)1.0 / temperature_, bottom_diff);
	}

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
		get_normalizer(normalization_, outer_num_*inner_num_);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxCrossEntropyMLoss);
#endif

INSTANTIATE_CLASS(SoftmaxCrossEntropyMLossLayer);
REGISTER_LAYER_CLASS(SoftmaxCrossEntropyMLoss);

}  // namespace caffe
