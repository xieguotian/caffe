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
	class AmplitudeTest : public MultiDeviceTest<TypeParam> 
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		AmplitudeTest() :blob_bottom_(new Blob<Dtype>(2, 4, 2, 2)),
			blob_top_(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~AmplitudeTest() { delete blob_bottom_; delete blob_top_; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(AmplitudeTest, TestDtypesAndDevices);

	TYPED_TEST(AmplitudeTest, TestForward)
	{
		typedef typename TypeParam::Dtype Dtype;
		caffe_set(blob_bottom_vec_[0]->count(), (Dtype)1, blob_bottom_vec_[0]->mutable_cpu_data());

		LayerParameter layer_param;
		AmplitudeLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num())
			<< "(top_num,bottom_num)=" << this->blob_top_->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_->channels(), 1)
			<< "(top_channels,bottom_channels)=" << this->blob_top_->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height())
			<< "(top_height,bottom_height)=" << this->blob_top_->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width())
			<< "(top_width,bottom_width)=" << this->blob_top_->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		for (int i = 0; i < blob_top_->count(); i++)
		{
			EXPECT_EQ(this->blob_top_->cpu_data()[i], 2)
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< 2;
		}
	}


	TYPED_TEST(AmplitudeTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.1);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(blob_bottom_);

		LayerParameter layer_param;
		AmplitudeLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}

}