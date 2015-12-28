#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/layers/non_local_2_layer.hpp"

namespace caffe {
	template <typename TypeParam>
	class Nonlocal2Test : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		Nonlocal2Test() :blob_bottom_(new Blob<Dtype>(2, 2, 5, 5)),
			blob_top_0(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_0);
		}

		virtual ~Nonlocal2Test() { delete blob_bottom_; delete blob_top_0; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_0;

		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(Nonlocal2Test, TestDtypesAndDevices);

	TYPED_TEST(Nonlocal2Test, TestForward)
	{
		typedef typename TypeParam::Dtype Dtype;
		//Dtype input[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };

		//caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());
		caffe_set(blob_bottom_->count(), (Dtype)0, blob_bottom_->mutable_cpu_data());
		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.1);
		sm_param->set_beta(10);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(5);
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		//conv_param->set_kernel_size(0,3);
		//conv_param->set_stride(0,1);
		//conv_param->set_pad(0,1);
		conv_param->add_kernel_size(3);
		conv_param->add_stride(1);
		conv_param->add_pad(1);
		NonLocal2Layer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_0->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_0->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_0->channels(),9)
			<< "(top_channels,bottom_channels)=" << this->blob_top_0->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_0->height(), 5)
			<< "(top_height,bottom_height)=" << this->blob_top_0->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_0->width(), 5)
			<< "(top_width,bottom_width)=" << this->blob_top_0->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_0->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_0->cpu_data()[i], 1.0/9.0, min_precision)
				<< "(top_data,gt_data)=" << this->blob_top_0->cpu_data()[i] << ","
				<< 0.5;
		}

	}

	TYPED_TEST(Nonlocal2Test, TestBackward)
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
		//conv_param->set_kernel_size(0,3);
		//conv_param->set_stride(0,2);
		//conv_param->set_pad(0,1);
		conv_param->add_kernel_size(3);
		conv_param->add_stride(2);
		conv_param->add_pad(1);
		NonLocal2Layer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}