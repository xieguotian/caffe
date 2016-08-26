#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxCrossEntropyLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
	 SoftmaxCrossEntropyLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 2, 3)),
		blob_bottom_hard_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand();
    }

	for (int i = 0; i < blob_bottom_hard_label_->count(); ++i) {
		blob_bottom_hard_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
	}

	int spat_dim = blob_bottom_label_->height()*blob_bottom_label_->width();
	for (int n = 0; n < blob_bottom_label_->num(); ++n)
	{
		for (int spat_idx = 0; spat_idx < spat_dim; ++spat_idx)
		{
			Dtype* st_pr = blob_bottom_label_->mutable_cpu_data() +
				n*blob_bottom_label_->channels()*spat_dim +
				spat_idx;

			Dtype sum_data = 0;
			for (int ch = 0; ch < blob_bottom_label_->channels(); ++ch)
			{
				sum_data += st_pr[ch*spat_dim];
			}
			for (int ch = 0; ch < blob_bottom_label_->channels(); ++ch)
			{
				st_pr[ch*spat_dim] /= sum_data;
			}
		}
	}
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
	 virtual ~SoftmaxCrossEntropyLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_hard_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxCrossEntropyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestIgnoreLabelGradient) {
	typedef typename TypeParam::Dtype Dtype;
	this->blob_bottom_vec_.push_back(blob_bottom_hard_label_);
	LayerParameter layer_param;
	layer_param.add_loss_weight(3);
	layer_param.mutable_loss_param()->set_ignore_label(3);
	SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_, 0);
}


TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestGradientTemperature) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	layer_param.add_loss_weight(3);
	layer_param.mutable_softmax_param()->set_temperature(8);
	SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestGradientIgnoreLabelTemperature) {
	typedef typename TypeParam::Dtype Dtype;
	this->blob_bottom_vec_.push_back(blob_bottom_hard_label_);
	LayerParameter layer_param;
	layer_param.add_loss_weight(3);
	layer_param.mutable_loss_param()->set_ignore_label(3);
	layer_param.mutable_softmax_param()->set_temperature(8);
	SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
	GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxCrossEntropyLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
