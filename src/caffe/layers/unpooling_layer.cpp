// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include <ctime>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void UnPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  max_top_blobs_ = 1;
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
	  CHECK(!(pool_param.has_kernel_size() ||
		  pool_param.has_kernel_h() || pool_param.has_kernel_w()))
		  << "With Global_pooling: true Filter size cannot specified";
  }
  else {
	  CHECK(!pool_param.has_kernel_size() !=
		  !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
		  << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
	  CHECK(pool_param.has_kernel_size() ||
		  (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
		  << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
	  && pool_param.has_pad_w())
	  || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
	  << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
	  && pool_param.has_stride_w())
	  || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
	  << "Stride is stride OR stride_h and stride_w are required.";

  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
	  kernel_h_ = bottom[0]->height();
	  kernel_w_ = bottom[0]->width();
  }
  else {
	  if (pool_param.has_kernel_size()) {
		  kernel_h_ = kernel_w_ = pool_param.kernel_size();
	  }
	  else {
		  kernel_h_ = pool_param.kernel_h();
		  kernel_w_ = pool_param.kernel_w();
	  }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
	  pad_h_ = pad_w_ = pool_param.pad();
  }
  else {
	  pad_h_ = pool_param.pad_h();
	  pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
	  stride_h_ = stride_w_ = pool_param.stride();
  }
  else {
	  stride_h_ = pool_param.stride_h();
	  stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
	  CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
		  << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
	  CHECK(this->layer_param_.pooling_param().pool()
		  == PoolingParameter_PoolMethod_AVE
		  || this->layer_param_.pooling_param().pool()
		  == PoolingParameter_PoolMethod_MAX)
		  << "Padding implemented only for average and max pooling.";
	  CHECK_LT(pad_h_, kernel_h_);
	  CHECK_LT(pad_w_, kernel_w_);
  }
  std::srand(std::time(0));
  
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
		<< "corresponding to (num, channels, height, width)";
	channels_ = bottom[0]->channels();
	pooled_height_ = bottom[0]->height();
	pooled_width_ = bottom[0]->width();

	height_ = static_cast<int>(ceil(static_cast<float>((pooled_height_ - 1) * stride_h_))) - 2 * pad_h_ + kernel_h_;
	width_ = static_cast<int>(ceil(static_cast<float>((pooled_width_ - 1) * stride_w_))) - 2 * pad_w_ + kernel_w_;

	if (bottom.size() == 3)
	{
		if (bottom[2]->height() - height_ < kernel_h_ &&
			bottom[2]->width() - width_ < kernel_w_)
		{
			height_ = bottom[2]->height();
			width_ = bottom[2]->width();
		}
	}
	if (global_pooling_)
	{
		kernel_h_ = height_;
		kernel_w_ = width_;
	}

	if (pad_h_ || pad_w_) {
		CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
		CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
	}
	top[0]->Reshape(bottom[0]->num(), channels_, height_,
		width_);
}

template <typename Dtype>
void UnPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //JTS DONE -> const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //JTS DONE ->  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_set(top[0]->count(), Dtype(0), top_data);
  // JTS done -> const Dtype* top_mask;
  const Dtype* bottom_mask;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // We require a bottom mask in position 1!
    assert(bottom.size() > 1);
    // The main loop
    bottom_mask = bottom[1]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index = bottom_mask[index];
            top_data[bottom_index] += bottom_data[index];
            //std::cout << "midx = " << bottom_index << " val: " << bottom_data[index] << " == " << top_data[bottom_index] << std::endl;
          }
        }
        top_data += top[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);

        bottom_mask += bottom[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[h * width_ + w] +=
                  bottom_data[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        top_data += top[0]->offset(0, 1);
        bottom_data += bottom[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
	  // The main loop
	  for (int n = 0; n < bottom[0]->num(); ++n) {
		  for (int c = 0; c < channels_; ++c) {
			  for (int ph = 0; ph < pooled_height_; ++ph) {
				  for (int pw = 0; pw < pooled_width_; ++pw) {
					  int hstart = ph * stride_h_ - pad_h_;
					  int wstart = pw * stride_w_ - pad_w_;
					  int hend = min(hstart + kernel_h_, height_);
					  int wend = min(wstart + kernel_w_, width_);
					  hstart = max(hstart, 0);
					  wstart = max(wstart, 0);
					  const int pool_index = ph * pooled_width_ + pw;
					  int h = std::rand() % (hend - hstart) + hstart;
					  int w = std::rand() % (wend - wstart) + wstart;
					  const int bottom_index = h*width_ + w;
					  //const int index = ph * pooled_width_ + pw;
					  //const int bottom_index = bottom_mask[index];
					  top_data[bottom_index] += bottom_data[pool_index];
					  //std::cout << "midx = " << bottom_index << " val: " << bottom_data[index] << " == " << top_data[bottom_index] << std::endl;
				  }
			  }
			  top_data += top[0]->offset(0, 1);
			  bottom_data += bottom[0]->offset(0, 1);
		  }
	  }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  const Dtype* top_mask;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    assert(bottom->size() > 1);
    top_mask = bottom[1]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index = top_mask[index];
            bottom_diff[index] += top_diff[bottom_index];
            //std::cout << "midx = " << bottom_index << " val: " << top_diff[bottom_index] << " == " << bottom_diff[index] << std::endl;
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        top_mask += bottom[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
      for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[ph * pooled_width_ + pw] +=
                  top_diff[h * width_ + w];
              }
            }
            bottom_diff[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(UnPoolingLayer);
#endif

INSTANTIATE_CLASS(UnPoolingLayer);
REGISTER_LAYER_CLASS(UnPooling);

}  // namespace caffe
