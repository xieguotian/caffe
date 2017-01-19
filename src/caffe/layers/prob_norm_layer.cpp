#include <algorithm>
#include <vector>

#include "caffe/layers/prob_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template<typename Dtype>
	void ProbNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		use_T_ = this->layer_param_.softmax_param().temperature() != 0;
		temperature_ = this->layer_param_.softmax_param().temperature();
	}

template <typename Dtype>
void ProbNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
  position_.Reshape(scale_dims);
}

template <typename Dtype>
void ProbNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_data;
	if (use_T_)
	{
		bottom_data = bottom[0]->mutable_cpu_diff();
		caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), bottom_data);
		caffe_scal(bottom[0]->count(), (Dtype)1.0 / temperature_, bottom_data);
	}
	else
	{
		bottom_data = bottom[0]->mutable_cpu_data();
	}
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }

  if (use_T_)
	  caffe_set(bottom[0]->count(), (Dtype)0.0, bottom[0]->mutable_cpu_diff());
}

template <typename Dtype>
void ProbNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
  if (use_T_)
	  caffe_scal(top[0]->count(), (Dtype)1.0 / temperature_, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(ProbNormLayer);
#endif

INSTANTIATE_CLASS(ProbNormLayer);
REGISTER_LAYER_CLASS(ProbNorm);
}  // namespace caffe
