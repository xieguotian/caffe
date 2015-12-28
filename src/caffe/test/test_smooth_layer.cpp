#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/smooth_threshold_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
	template <typename TypeParam>
	class SmoothTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		SmoothTest() :blob_bottom_(new Blob<Dtype>(2, 3, 2, 2)),
			blob_top_(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~SmoothTest() { delete blob_bottom_; delete blob_top_; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(SmoothTest, TestDtypesAndDevices);

	TYPED_TEST(SmoothTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 3, 6, 4, 14, 12, 15, 13, 18,
			16, 7, 23, 17, 0, 22, 21, 8, 1, 11, 10, 19, 5, 9, 2, 20 };

		Dtype output[] = { 2.06115358e-08, 9.99995460e-01, 4.53793276e-04, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.92874985e-21, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			4.24835426e-17, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			9.09090909e-01, 1.00000000e+00, 9.35762297e-13, 1.00000000e+00 };
		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.1);
		sm_param->set_beta(10);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(5);
		SmoothThresholdLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(), blob_bottom_->channels())
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width())
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

	TYPED_TEST(SmoothTest, TestForward_2)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 15, 17, 0, 13, 4, 3, 2, 20, 16, 21, 6, 10,
			22, 8, 9, 23, 7, 1, 14, 5, 12, 18, 19, 11 };

		Dtype output[] = { 1.00000000e+00, 1.00000000e+00, 1.92874985e-21, 1.00000000e+00,
			4.53793276e-04, 2.06115358e-08, 9.35762297e-13, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 9.99995460e-01, 1.00000000e+00,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
			1.00000000e+00, 4.24835426e-17, 1.00000000e+00, 9.09090909e-01,
			1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00 };
		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.1);
		sm_param->set_beta(10);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(5);
		SmoothThresholdLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(), blob_bottom_->channels())
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width())
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

	TYPED_TEST(SmoothTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.5);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(blob_bottom_);
		LayerParameter layer_param;
		SmoothThresholdParameter* sm_param = layer_param.mutable_smooth_threshold_param();
		sm_param->set_alpha(0.01);
		sm_param->set_beta(100);
		sm_param->mutable_threshold_filler()->set_type("constant");
		sm_param->mutable_threshold_filler()->set_value(0.3);
		SmoothThresholdLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-5, 1e-3, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}

}