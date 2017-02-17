#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_param.mutable_loss_weight()->Clear();
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

  LayerParameter softmax_param_org(this->layer_param_);
  softmax_param_org.set_type("Softmax");
  softmax_param_org.mutable_loss_weight()->Clear();
  softmax_param_org.mutable_softmax_param()->set_temperature(1);

  softmax_layer_org_ = LayerRegistry<Dtype>::CreateLayer(softmax_param_org);
  softmax_bottom_vec_org_.clear();
  softmax_bottom_vec_org_.push_back(bottom[0]);
  softmax_top_vec_org_.clear();
  softmax_top_vec_org_.push_back(&prob_org_);
  softmax_layer_org_->SetUp(softmax_bottom_vec_org_, softmax_top_vec_org_);

  softmax_bottom_vec_label_.clear();
  softmax_bottom_vec_label_.push_back(bottom[1]);
  softmax_top_vec_label_.clear();
  softmax_top_vec_label_.push_back(&label_);

  is_update_T_ = false;
  if (this->layer_param_.param_size() > 0)
  {
	  this->blobs_.resize(1);
	  vector<int> sz;
	  sz.push_back(1);
	  this->blobs_[0].reset(new Blob<Dtype>(sz));
	  this->blobs_[0]->mutable_cpu_data()[0] = temperature_;
	  update_step_ = this->layer_param_.param(0).decay_mult();
	  is_update_T_ = true;
  }

}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);

  if (is_update_T_ && top.size() >= 2)
  {
	  top[1]->ReshapeLike(*this->blobs_[0]);
  }
  else
  {
	  if (top.size() >= 2) {
		  // softmax output
		  top[1]->ReshapeLike(*bottom[0]);
	  }
  }
  softmax_layer_org_->Reshape(softmax_bottom_vec_org_, softmax_top_vec_org_);
  softmax_top_vec_label_[0]->ReshapeLike(*softmax_top_vec_[0]);
  if (is_update_T_)
  {
	  boost::dynamic_pointer_cast<SoftmaxLayer<Dtype> >(softmax_layer_)->set_temperture(this->blobs_[0]->cpu_data()[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxCrossEntropyLossLayer<Dtype>::get_normalizer(
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
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_cpu(
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
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
STUB_GPU(SoftmaxCrossEntropyLoss);
#endif

INSTANTIATE_CLASS(SoftmaxCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SoftmaxCrossEntropyLoss);

}  // namespace caffe
