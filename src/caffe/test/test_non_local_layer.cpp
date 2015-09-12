#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
	template <typename TypeParam>
	class NonlocalTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		NonlocalTest() :blob_bottom_(new Blob<Dtype>(2, 2, 2, 2)),
			 blob_top_(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~NonlocalTest() { delete blob_bottom_; delete blob_top_; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(NonlocalTest, TestDtypesAndDevices);

	TYPED_TEST(NonlocalTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };

		Dtype output[] = { 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.50000000e+01, 0.00000000e+00, 0.00000000e+00,
			1.50000000e+01, 1.30000000e+01, 0.00000000e+00,
			0.00000000e+00, 1.18181818e+01, 0.00000000e+00,
			0.00000000e+00, 1.49999995e+01, 0.00000000e+00,
			1.20000000e+01, 2.89312477e-20, 2.50737480e-20,
			2.31449982e-20, 0.00000000e+00, 1.29999996e+01,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.09090909e+01, 0.00000000e+00,
			0.00000000e+00, 1.20000000e+01, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			5.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			5.00000000e+00, 8.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 7.27272727e+00, 0.00000000e+00,
			0.00000000e+00, 4.99999985e+00, 0.00000000e+00,
			1.10000000e+01, 9.64374924e-21, 1.54299988e-20,
			2.12162483e-20, 7.71499939e-21, 7.99999976e+00,
			0.00000000e+00, 4.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.00000000e+01, 0.00000000e+00,
			0.00000000e+00, 1.10000000e+01, 4.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 4.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.40000000e+01, 0.00000000e+00, 0.00000000e+00,
			1.40000000e+01, 6.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 6.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.40000000e+01, 0.00000000e+00,
			7.00000000e+00, 2.70024979e-20, 1.15724991e-20,
			1.35012489e-20, 5.78624954e-21, 6.00000000e+00,
			0.00000000e+00, 3.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 7.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 7.00000000e+00, 3.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 3.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.00000000e+01, 0.00000000e+00, 0.00000000e+00,
			1.00000000e+01, 9.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 9.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.00000000e+01, 0.00000000e+00,
			1.00000000e+00, 1.92874985e-20, 1.73587486e-20,
			1.92874985e-21, 3.85749970e-21, 9.00000000e+00,
			0.00000000e+00, 2.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.00000000e+00, 2.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 2.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00 };
		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.1);
		sm_param->set_beta(10);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(5);
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->set_kernel_size(3);
		conv_param->set_stride(1);
		conv_param->set_pad(1);
		NonLocalLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(),18)
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_->cpu_data()[i], output[i], min_precision)
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< output[i];
		}
	}

	TYPED_TEST(NonlocalTest, TestForward_2)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 9, 6, 11, 21, 31, 30, 25, 28, 2, 26, 17, 18, 0, 34, 22, 29, 5,
			27, 15, 12, 16, 19, 8, 1, 4, 33, 3, 35, 20, 14, 32, 24, 13, 7,
			23, 10 };

		Dtype output[] = { 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			3.10000000e+01, 0.00000000e+00, 0.00000000e+00,
			2.10000000e+01, 3.00000000e+01, 0.00000000e+00,
			0.00000000e+00, 3.10000000e+01, 0.00000000e+00,
			0.00000000e+00, 6.00000000e+00, 0.00000000e+00,
			2.80000000e+01, 1.73587486e-20, 2.12162483e-20,
			4.82187462e-20, 3.85749970e-21, 6.00000000e+00,
			0.00000000e+00, 2.80000000e+01, 0.00000000e+00,
			0.00000000e+00, 3.10000000e+01, 0.00000000e+00,
			0.00000000e+00, 2.10000000e+01, 3.00000000e+01,
			0.00000000e+00, 0.00000000e+00, 3.10000000e+01,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			3.40000000e+01, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 2.20000000e+01, 0.00000000e+00,
			0.00000000e+00, 3.40000000e+01, 0.00000000e+00,
			0.00000000e+00, 1.70000000e+01, 0.00000000e+00,
			5.00000000e+00, 5.01474960e-20, 3.47174973e-20,
			5.59337456e-20, 5.20762459e-20, 1.70000000e+01,
			0.00000000e+00, 5.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 3.40000000e+01, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 2.20000000e+01,
			0.00000000e+00, 0.00000000e+00, 3.40000000e+01,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			8.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.90000000e+01, 9.99999969e-01, 0.00000000e+00,
			0.00000000e+00, 8.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.20000000e+01, 0.00000000e+00,
			3.30000000e+01, 2.89312477e-20, 3.08599976e-20,
			7.71499939e-21, 5.78624954e-21, 1.20000000e+01,
			0.00000000e+00, 3.30000000e+01, 0.00000000e+00,
			0.00000000e+00, 8.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.90000000e+01, 1.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 8.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			2.40000000e+01, 0.00000000e+00, 0.00000000e+00,
			3.20000000e+01, 1.29999996e+01, 0.00000000e+00,
			0.00000000e+00, 2.40000000e+01, 0.00000000e+00,
			0.00000000e+00, 2.00000000e+01, 0.00000000e+00,
			2.30000000e+01, 6.75062447e-20, 2.70024979e-20,
			1.35012489e-20, 1.92874985e-20, 2.00000000e+01,
			0.00000000e+00, 2.30000000e+01, 0.00000000e+00,
			0.00000000e+00, 2.40000000e+01, 0.00000000e+00,
			0.00000000e+00, 3.20000000e+01, 1.30000000e+01,
			0.00000000e+00, 0.00000000e+00, 2.40000000e+01,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00 };
		blob_bottom_->Reshape(2, 2, 3, 3);
		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.1);
		sm_param->set_beta(10);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(5);
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->set_kernel_size(3);
		conv_param->set_stride(2);
		conv_param->set_pad(1);
		NonLocalLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(), 18)
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_->cpu_data()[i], output[i], min_precision)
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< output[i];
		}
	}

	TYPED_TEST(NonlocalTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.5);
		GaussianFiller<Dtype> filler(filler_param);
		blob_bottom_->Reshape(2, 2, 3, 3);
		filler.Fill(blob_bottom_);
		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.01);
		sm_param->set_beta(100);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(0.3);

		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->set_kernel_size(3);
		conv_param->set_stride(2);
		conv_param->set_pad(1);
		NonLocalLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-5, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}