#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_tree_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifdef USE_CUDNN

template <typename Dtype>
class CuDNNConvolutionTreeLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNConvolutionTreeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 16, 1, 1)),
        blob_bottom_2_(new Blob<Dtype>(2, 16, 1, 1)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CuDNNConvolutionTreeLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNConvolutionTreeLayerTest, TestDtypes);

TYPED_TEST(CuDNNConvolutionTreeLayerTest, TestGradientCuDNN_4channels) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(16);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNConvolutionTreeLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
TYPED_TEST(CuDNNConvolutionTreeLayerTest, TestGradientCuDNN_4channels_up) {
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->set_num_output(64);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("gaussian");
	CuDNNConvolutionTreeLayer<TypeParam> layer(layer_param);
	GradientChecker<TypeParam> checker(1e-2, 1e-3);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}
TYPED_TEST(CuDNNConvolutionTreeLayerTest, TestGradientCuDNN_4channels_down) {
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->set_num_output(4);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("gaussian");
	CuDNNConvolutionTreeLayer<TypeParam> layer(layer_param);
	GradientChecker<TypeParam> checker(1e-2, 1e-3);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}
TYPED_TEST(CuDNNConvolutionTreeLayerTest, TestGradientCuDNN_4channels_shuffle) {
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->set_num_output(16);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("gaussian");
	convolution_param->set_shuffle(true);
	CuDNNConvolutionTreeLayer<TypeParam> layer(layer_param);
	GradientChecker<TypeParam> checker(1e-2, 1e-3);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_,0);
	CuDNNConvolutionTreeLayer<TypeParam> layer2(layer_param);
	checker.CheckGradientExhaustive(&layer2, this->blob_bottom_vec_,
		this->blob_top_vec_, 1);
}

TYPED_TEST(CuDNNConvolutionTreeLayerTest, TestGradientCuDNN_8channels) {
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->set_num_output(16);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("gaussian");
	convolution_param->set_num_channels_per_supernode(8);
	CuDNNConvolutionTreeLayer<TypeParam> layer(layer_param);
	GradientChecker<TypeParam> checker(1e-2, 1e-3);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}

TYPED_TEST(CuDNNConvolutionTreeLayerTest, TestGradientCuDNN_4channels_tree_2) {
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->set_num_output(16);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("gaussian");
	convolution_param->set_num_layer_of_tree(1);
	CuDNNConvolutionTreeLayer<TypeParam> layer(layer_param);
	GradientChecker<TypeParam> checker(1e-2, 1e-3);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}
#endif

}  // namespace caffe
