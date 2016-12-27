#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/entropy_total_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class EntropyTotalWithLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
	 EntropyTotalWithLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
	 virtual ~EntropyTotalWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EntropyTotalWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(EntropyTotalWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  EntropyTotalWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(EntropyTotalWithLossLayerTest, TestGradient2) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	layer_param.add_loss_weight(3);
	EntropyTotalWithLossLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	this->blob_bottom_vec_[0] = new Blob<Dtype>(10, 5, 2, 3);
	this->blob_bottom_vec_[1] = new Blob<Dtype>(10, 1, 2, 3);
	// fill the values
	FillerParameter filler_param;
	filler_param.set_std(10);
	GaussianFiller<Dtype> filler(filler_param);
	filler.Fill(this->blob_bottom_vec_[0]);
	filler.Fill(this->blob_bottom_vec_[1]);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_, 0);
}

TYPED_TEST(EntropyTotalWithLossLayerTest, TestGradient3) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	layer_param.add_loss_weight(3);
	layer_param.mutable_softmax_param()->set_temperature(4);
	EntropyTotalWithLossLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	this->blob_bottom_vec_[0] = new Blob<Dtype>(10, 5, 2, 3);
	this->blob_bottom_vec_[1] = new Blob<Dtype>(10, 1, 2, 3);
	// fill the values
	FillerParameter filler_param;
	filler_param.set_std(10);
	GaussianFiller<Dtype> filler(filler_param);
	filler.Fill(this->blob_bottom_vec_[0]);
	filler.Fill(this->blob_bottom_vec_[1]); 
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_, 0);
}
}  // namespace caffe
