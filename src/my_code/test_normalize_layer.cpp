#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
	template <typename TypeParam>
	class NormalizeLayerTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		NormalizeLayerTest() :blob_bottom_(new Blob<Dtype>(2, 4, 3, 3)),
			blob_top_(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~NormalizeLayerTest() 
		{ 
			delete blob_bottom_; 
			delete blob_top_; 
		}


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

	TYPED_TEST(NormalizeLayerTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		caffe_set(this->blob_bottom_->count(), (Dtype)1, this->blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;

		NormalizeLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels())
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width())
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_->width();
		
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype min_precision = 1e-5;
		
		for (int i = 0; i < this->blob_top_->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_->cpu_data()[i], 0.25, min_precision);
		}
		
	}

	TYPED_TEST(NormalizeLayerTest, TestForward_2)
	{
		typedef typename TypeParam::Dtype Dtype;
		caffe_set(this->blob_bottom_->count(), (Dtype)1.2, this->blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;

		NormalizeLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels())
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width())
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < this->blob_top_->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_->cpu_data()[i], 0.25, min_precision);
		}
	}

	TYPED_TEST(NormalizeLayerTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.5);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom_);

		LayerParameter layer_param;

		NormalizeLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}