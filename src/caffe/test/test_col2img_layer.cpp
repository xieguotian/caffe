#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
	template <typename TypeParam>
	class Col2ImgMaskLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		Col2ImgMaskLayerTest()
			: blob_bottom_(new Blob<Dtype>(2, 18, 2, 2)),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~Col2ImgMaskLayerTest() { delete blob_bottom_; delete blob_top_; }
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(Col2ImgMaskLayerTest, TestDtypesAndDevices);

	TYPED_TEST(Col2ImgMaskLayerTest, TestForward_1) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* convolution_param =
			layer_param.mutable_convolution_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		caffe_set(blob_bottom_->count(), (Dtype)3, blob_bottom_->mutable_cpu_data());
		Col2imgMaskLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);

		EXPECT_EQ(this->blob_top_->num(), 2);
		EXPECT_EQ(this->blob_top_->channels(), 2);
		EXPECT_EQ(this->blob_top_->height(), 5);
		EXPECT_EQ(this->blob_top_->width(), 5);

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_->count(); i++)
			EXPECT_NEAR(blob_top_->mutable_cpu_data()[i], 3, min_precision);
	}

	TYPED_TEST(Col2ImgMaskLayerTest, TestForward_2) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* convolution_param =
			layer_param.mutable_convolution_param();
		convolution_param->set_kernel_size(3);
		convolution_param->set_stride(2);
		caffe_set(blob_bottom_->count(), (Dtype)2, blob_bottom_->mutable_cpu_data());

		Blob<Dtype> mask;
		mask.ReshapeLike(*blob_bottom_);
		caffe_set(mask.count(), (Dtype)1, mask.mutable_cpu_data());
		vector<Blob<Dtype>*> blob_bottom_vec_2_;
		blob_bottom_vec_2_.push_back(blob_bottom_);
		blob_bottom_vec_2_.push_back(&mask);

		Col2imgMaskLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_2_, blob_top_vec_);

		EXPECT_EQ(this->blob_top_->num(), 2);
		EXPECT_EQ(this->blob_top_->channels(), 2);
		EXPECT_EQ(this->blob_top_->height(), 5);
		EXPECT_EQ(this->blob_top_->width(), 5);

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_->count(); i++)
			EXPECT_NEAR(blob_top_->mutable_cpu_data()[i], 2, min_precision);
	}

	TYPED_TEST(Col2ImgMaskLayerTest, TestBackward) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* convolution_param =
			layer_param.mutable_convolution_param();
		convolution_param->set_kernel_h(3);
		convolution_param->set_kernel_w(3);
		convolution_param->set_stride(2);
		Col2imgMaskLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-2);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}