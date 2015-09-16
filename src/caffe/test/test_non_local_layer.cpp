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
			blob_top_0(new Blob<Dtype>()), blob_top_1(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_0);
			blob_top_vec_.push_back(blob_top_1);
			blob_top_vec_2.push_back(blob_top_0);
		}

		virtual ~NonlocalTest() { delete blob_bottom_; delete blob_top_0; delete blob_top_1; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_0;
		Blob<Dtype>* const blob_top_1;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		vector<Blob<Dtype>*> blob_top_vec_2;
	};

	TYPED_TEST_CASE(NonlocalTest, TestDtypesAndDevices);

	TYPED_TEST(NonlocalTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };

		Dtype output_0[] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.38879439e-10, 4.53793276e-04,
			1.00000000e+00, 2.51099853e-07, 1.00000000e+00,
			1.00000000e+00, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 4.53793276e-04,
			1.00000000e+00, 1.38879439e-10, 1.00000000e+00,
			4.53793276e-04, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 4.53793276e-04, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 4.53793276e-04,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.38879439e-10, 4.53793276e-04,
			1.00000000e+00, 2.51099853e-07, 1.00000000e+00,
			1.00000000e+00, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 4.53793276e-04,
			1.00000000e+00, 1.38879439e-10, 1.00000000e+00,
			4.53793276e-04, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 4.53793276e-04, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 4.53793276e-04,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			5.50042173e-03, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 1.00000000e+00,
			1.00000000e+00, 5.50042173e-03, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			2.51099853e-07, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 2.51099853e-07, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			5.50042173e-03, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 1.00000000e+00,
			1.00000000e+00, 5.50042173e-03, 2.51099853e-07,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			2.51099853e-07, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 2.51099853e-07, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 2.51099853e-07 };
		Dtype output_1[] = { 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.50000000e+01, 0.00000000e+00, 0.00000000e+00,
			1.50000000e+01, 1.30000000e+01, 0.00000000e+00,
			0.00000000e+00, 1.80543270e-09, 0.00000000e+00,
			0.00000000e+00, 3.76649779e-06, 0.00000000e+00,
			1.20000000e+01, 2.89312477e-20, 2.50737480e-20,
			2.31449982e-20, 0.00000000e+00, 3.26429808e-06,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.66655326e-09, 0.00000000e+00,
			0.00000000e+00, 1.20000000e+01, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			5.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			5.00000000e+00, 8.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.11103551e-09, 0.00000000e+00,
			0.00000000e+00, 1.25549926e-06, 0.00000000e+00,
			1.10000000e+01, 9.64374924e-21, 1.54299988e-20,
			2.12162483e-20, 7.71499939e-21, 2.00879882e-06,
			0.00000000e+00, 4.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.52767382e-09, 0.00000000e+00,
			0.00000000e+00, 1.10000000e+01, 4.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 4.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.40000000e+01, 0.00000000e+00, 0.00000000e+00,
			1.40000000e+01, 6.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 6.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.40000000e+01, 0.00000000e+00,
			3.85029521e-02, 2.70024979e-20, 1.15724991e-20,
			1.35012489e-20, 5.78624954e-21, 6.00000000e+00,
			0.00000000e+00, 1.65012652e-02, 0.00000000e+00,
			0.00000000e+00, 7.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 7.00000000e+00, 3.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 3.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
			1.00000000e+01, 0.00000000e+00, 0.00000000e+00,
			1.00000000e+01, 9.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 9.00000000e+00, 0.00000000e+00,
			0.00000000e+00, 1.00000000e+01, 0.00000000e+00,
			5.50042173e-03, 1.92874985e-20, 1.73587486e-20,
			1.92874985e-21, 3.85749970e-21, 9.00000000e+00,
			0.00000000e+00, 1.10008435e-02, 0.00000000e+00,
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
		EXPECT_EQ(this->blob_top_0->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_0->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_0->channels(),18)
			<< "(top_channels,bottom_channels)=" << this->blob_top_0->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_0->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_0->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_0->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_0->width() << ","
			<< this->blob_bottom_->width();

		EXPECT_EQ(this->blob_top_1->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_1->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_1->channels(), 18)
			<< "(top_channels,bottom_channels)=" << this->blob_top_1->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_1->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_1->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_1->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_1->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_0->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_0->cpu_data()[i], output_0[i], min_precision)
				<< "(top_data,gt_data)=" << this->blob_top_0->cpu_data()[i] << ","
				<< output_0[i];
		}

		for (int i = 0; i < blob_top_1->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_1->cpu_data()[i], output_1[i], min_precision)
				<< "(top_data,gt_data)=" << this->blob_top_1->cpu_data()[i] << ","
				<< output_1[i];
		}
	}

	TYPED_TEST(NonlocalTest, TestForward_2)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 13, 3, 5, 8, 12, 27, 32, 24, 4, 10, 34, 31, 6, 33, 14, 16, 18,
			25, 21, 19, 30, 28, 20, 1, 23, 7, 0, 9, 29, 22, 35, 17, 26, 15,
			11, 2 };

		Dtype output[] = { 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 2.51099853e-07, 1.00000000e+00,
			1.00000000e+00, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 2.51099853e-07, 1.00000000e+00,
			1.00000000e+00, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 2.51099853e-07, 4.24835426e-17,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 4.24835426e-17,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			4.24835426e-17, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 4.24835426e-17, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 4.24835426e-17,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 2.51099853e-07, 4.24835426e-17,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.92874985e-21, 1.92874985e-21,
			1.92874985e-21, 1.92874985e-21, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 4.24835426e-17,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			4.24835426e-17, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 4.24835426e-17, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 4.24835426e-17 };
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
		layer.SetUp(blob_bottom_vec_, blob_top_vec_2);
		EXPECT_EQ(this->blob_top_0->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_0->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_0->channels(), 18)
			<< "(top_channels,bottom_channels)=" << this->blob_top_0->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_0->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_0->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_0->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_0->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_2);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_0->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_0->cpu_data()[i], output[i], min_precision)
				<< "(top_data,gt_data)=" << this->blob_top_0->cpu_data()[i] << ","
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