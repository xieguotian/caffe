#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_torch_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define INPUT_DATA_SIZE 3

namespace caffe {

  template <typename TypeParam>
  class BatchNormTorchLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
   protected:
	   BatchNormTorchLayerTest()
        : blob_bottom_(new Blob<Dtype>(5, 2, 3, 4)),
          blob_top_(new Blob<Dtype>()) {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
    }
	   virtual ~BatchNormTorchLayerTest() { delete blob_bottom_; delete blob_top_; }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
  };

  TYPED_TEST_CASE(BatchNormTorchLayerTest, TestDtypesAndDevices);

  TYPED_TEST(BatchNormTorchLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

	BatchNormTorchLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Test mean
    int num = this->blob_bottom_->num();
    int channels = this->blob_bottom_->channels();
    int height = this->blob_bottom_->height();
    int width = this->blob_bottom_->width();

    for (int j = 0; j < channels; ++j) {
      Dtype sum = 0, var = 0;
      for (int i = 0; i < num; ++i) {
        for ( int k = 0; k < height; ++k ) {
          for ( int l = 0; l < width; ++l ) {
            Dtype data = this->blob_top_->data_at(i, j, k, l);
            sum += data;
			std::cout<< data;
            var += data * data;
          }
        }
      }
      sum /= height * width * num;
      var /= height * width * num;

      const Dtype kErrorBound = 0.001;
      // expect zero mean
      EXPECT_NEAR(0, sum, kErrorBound);
      // expect unit variance
      EXPECT_NEAR(1, var, kErrorBound);
    }
  }

  TYPED_TEST(BatchNormTorchLayerTest, TestForwardGlobal) {
	  typedef typename TypeParam::Dtype Dtype;
	  LayerParameter layer_param;
	  layer_param.mutable_batch_norm_param()->set_use_global_stats(true);

	  layer_param.mutable_scale_param()->set_bias_term(true);
	  layer_param.mutable_scale_param()->mutable_filler()->set_type("constant");
	  layer_param.mutable_scale_param()->mutable_filler()->set_value(0.5);
	  layer_param.mutable_scale_param()->mutable_bias_filler()->set_type("constant");
	  layer_param.mutable_scale_param()->mutable_bias_filler()->set_value(1);

	  BatchNormTorchLayer<Dtype> layer(layer_param);
	  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	  // Test mean
	  int num = this->blob_bottom_->num();
	  int channels = this->blob_bottom_->channels();
	  int height = this->blob_bottom_->height();
	  int width = this->blob_bottom_->width();

	  for (int j = 0; j < channels; ++j) {
		  Dtype sum = 0, var = 0;
		  for (int i = 0; i < num; ++i) {
			  for (int k = 0; k < height; ++k) {
				  for (int l = 0; l < width; ++l) {
					  Dtype data = this->blob_top_->data_at(i, j, k, l);
					  sum += data;
					  std::cout << data;
					  var += (data - 1) * (data - 1);
				  }
			  }
		  }
		  sum /= height * width * num;
		  var /= height * width * num;

		  const Dtype kErrorBound = 0.001;
		  // expect zero mean
		  EXPECT_NEAR(0 * 0.5 + 1, sum, kErrorBound);
		  // expect unit variance
		  EXPECT_NEAR(0.5*0.5, var, kErrorBound);
	  }
  }

  TYPED_TEST(BatchNormTorchLayerTest, TestForwardScale) {
	  typedef typename TypeParam::Dtype Dtype;
	  LayerParameter layer_param;
	  layer_param.mutable_scale_param()->set_bias_term(true);
	  layer_param.mutable_scale_param()->mutable_filler()->set_type("constant");
	  layer_param.mutable_scale_param()->mutable_filler()->set_value(0.5);
	  layer_param.mutable_scale_param()->mutable_bias_filler()->set_type("constant");
	  layer_param.mutable_scale_param()->mutable_bias_filler()->set_value(1);
	  BatchNormTorchLayer<Dtype> layer(layer_param);
	  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	  // Test mean
	  int num = this->blob_bottom_->num();
	  int channels = this->blob_bottom_->channels();
	  int height = this->blob_bottom_->height();
	  int width = this->blob_bottom_->width();

	  for (int j = 0; j < channels; ++j) {
		  Dtype sum = 0, var = 0;
		  for (int i = 0; i < num; ++i) {
			  for (int k = 0; k < height; ++k) {
				  for (int l = 0; l < width; ++l) {
					  Dtype data = this->blob_top_->data_at(i, j, k, l);
					  sum += data;
					  std::cout << data;
					  var += (data-1) * (data-1);
				  }
			  }
		  }
		  sum /= height * width * num;
		  var /= height * width * num;

		  const Dtype kErrorBound = 0.001;
		  // expect zero mean
		  EXPECT_NEAR(0*0.5+1, sum, kErrorBound);
		  // expect unit variance
		  EXPECT_NEAR(0.5*0.5, var, kErrorBound);
	  }
  }
  TYPED_TEST(BatchNormTorchLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;

	BatchNormTorchLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-4);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }
  TYPED_TEST(BatchNormTorchLayerTest, TestGradientScale) {
	  typedef typename TypeParam::Dtype Dtype;
	  LayerParameter layer_param;
	  layer_param.mutable_scale_param()->set_bias_term(true);
	  layer_param.mutable_scale_param()->mutable_filler()->set_type("constant");
	  layer_param.mutable_scale_param()->mutable_filler()->set_value(0.5);
	  layer_param.mutable_scale_param()->mutable_bias_filler()->set_type("constant");
	  layer_param.mutable_scale_param()->mutable_bias_filler()->set_value(1);
	  BatchNormTorchLayer<Dtype> layer(layer_param);
	  GradientChecker<Dtype> checker(1e-2, 1e-4);
	  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		  this->blob_top_vec_);
  }
}  // namespace caffe
