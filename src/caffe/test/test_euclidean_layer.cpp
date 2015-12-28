#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
	template <typename TypeParam>
	class EuclideanTest : public MultiDeviceTest<TypeParam> 
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		EuclideanTest() :blob_bottom_0_(new Blob<Dtype>(2, 3, 2, 2)),
			blob_bottom_1_(new Blob<Dtype>(2, 3, 2, 2)),blob_top_(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_0_);
			blob_bottom_vec_.push_back(blob_bottom_1_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~EuclideanTest() { delete blob_bottom_0_;delete blob_bottom_1_; delete blob_top_; }


		Blob<Dtype>* const blob_bottom_0_;
		Blob<Dtype>* const blob_bottom_1_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(EuclideanTest, TestDtypesAndDevices);

	TYPED_TEST(EuclideanTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input_0[] = { 17, 16, 2, 23, 19, 5, 8, 7, 22, 1, 9,
			18, 12, 14, 21, 11, 13, 6, 4, 3, 10, 20, 0, 15 };
		Dtype input_1[] = { 10, 6, 8, 13, 12, 15, 1, 21, 20, 14, 5,
			16, 9, 23, 22, 18, 3, 17, 19, 11, 0, 2, 7, 4 };
		Dtype output[] = { 102 / 2.0, 369 / 2.0, 101 / 2.0, 300 / 2.0, 209 / 2.0, 526 / 2.0, 275 / 2.0, 234 / 2.0 };
		caffe_copy(blob_bottom_0_->count(), input_0, blob_bottom_0_->mutable_cpu_data());
		caffe_copy(blob_bottom_1_->count(), input_1, blob_bottom_1_->mutable_cpu_data());

		LayerParameter layer_param;
		EuclideanLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_0_->num();
		EXPECT_EQ(this->blob_top_->channels(), 1)
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_0_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_0_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width())
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_0_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		for (int i = 0; i < blob_top_->count(); i++)
		{
			EXPECT_EQ(this->blob_top_->cpu_data()[i], output[i])
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< output[i];
		}
	}

	TYPED_TEST(EuclideanTest, TestForward_2)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input_0[] = { 12, 9, 4, 17, 0, 8, 6, 19, 11,
			20, 14, 13, 3, 10, 1, 15, 18, 21, 7, 5, 22, 16, 23, 2 };
		Dtype input_1[] = { 17, 2, 23, 5, 20, 6, 1, 12, 7,
			15, 14, 3, 4, 11, 13, 22, 18, 16, 19, 8, 0, 9, 10, 21 };
		Dtype output[] = { 441 / 2.0, 78 / 2.0, 386 / 2.0, 293 / 2.0, 485 / 2.0, 75 / 2.0, 457 / 2.0, 419 / 2.0 };
		caffe_copy(blob_bottom_0_->count(), input_0, blob_bottom_0_->mutable_cpu_data());
		caffe_copy(blob_bottom_1_->count(), input_1, blob_bottom_1_->mutable_cpu_data());

		LayerParameter layer_param;
		EuclideanLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_0_->num();
		EXPECT_EQ(this->blob_top_->channels(), 1)
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_0_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_0_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0_->width())
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_0_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		for (int i = 0; i < blob_top_->count(); i++)
		{
			EXPECT_EQ(this->blob_top_->cpu_data()[i], output[i])
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< output[i];
		}
	}

	TYPED_TEST(EuclideanTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.1);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(blob_bottom_0_);
		filler.Fill(blob_bottom_1_);
		LayerParameter layer_param;
		EuclideanLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}

}