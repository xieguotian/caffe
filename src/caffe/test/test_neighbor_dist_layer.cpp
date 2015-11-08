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
	class NeighborDistLayerTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		NeighborDistLayerTest() :blob_bottom_(new Blob<Dtype>(2, 2, 2, 2)),
			blob_top_0(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_0);
			
		}

		virtual ~NeighborDistLayerTest() { delete blob_bottom_; delete blob_top_0; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_0;

		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(NeighborDistLayerTest, TestDtypesAndDevices);

	TYPED_TEST(NeighborDistLayerTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };
		//Dtype output[] = { 250., 233., 265., 226., 250., 233., 45., 185., 250.,
		//	233., 10., 16., 250., 13., 265., 193., 0., 0.,
		//	0., 0., 13., 233., 193., 16., 250., 10., 265.,
		//	16., 45., 185., 265., 16., 226., 233., 265., 16.,
		//	296., 117., 50., 185., 296., 117., 130., 58., 296.,
		//	117., 65., 13., 296., 65., 50., 17., 0., 0.,
		//	0., 0., 65., 117., 17., 13., 296., 65., 50.,
		//	13., 130., 58., 50., 13., 185., 117., 50., 13. };
		Dtype output[] = { -1., -1., -1., 15.03329638,
			-1., -1., 6.70820393, 13.60147051,
			-1., -1., 3.16227766, -1.,
			-1., 3.60555128, -1., 13.89244399,
			0., 0., 0., 0.,
			3.60555128, -1., 13.89244399, -1.,
			-1., 3.16227766, -1., -1.,
			6.70820393, 13.60147051, -1., -1.,
			15.03329638, -1., -1., -1.,
			-1., -1., -1., 13.60147051,
			-1., -1., 11.40175425, 7.61577311,
			-1., -1., 8.06225775, -1.,
			-1., 8.06225775, -1., 4.12310563,
			0., 0., 0., 0.,
			8.06225775, -1., 4.12310563, -1.,
			-1., 8.06225775, -1., -1.,
			11.40175425, 7.61577311, -1., -1.,
			13.60147051, -1., -1., -1. };
		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;

		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(3);
		conv_param->add_stride(1);
		conv_param->add_pad(1);
		NeighborDistLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_0->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_0->num() << ","
			<< this->blob_bottom_->num();
		EXPECT_EQ(this->blob_top_0->channels(),9)
			<< "(top_channels,bottom_channels)=" << this->blob_top_0->channels() << ","
			<< this->blob_bottom_->channels();
		EXPECT_EQ(this->blob_top_0->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_0->height() << ","
			<< this->blob_bottom_->height();
		EXPECT_EQ(this->blob_top_0->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_0->width() << ","
			<< this->blob_bottom_->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_0->count(); i++)
		{
			if (output[i]>=0)
			{
				EXPECT_NEAR(this->blob_top_0->cpu_data()[i], output[i], min_precision)
					<< "(top_data,gt_data)=" << this->blob_top_0->cpu_data()[i] << ","
					<< output[i];
			}
		}
	}

	TYPED_TEST(NeighborDistLayerTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.5);
		GaussianFiller<Dtype> filler(filler_param);
		blob_bottom_->Reshape(2, 2, 3, 5);
		filler.Fill(blob_bottom_);
		//Dtype input[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };
		//caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());
		LayerParameter layer_param;

		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(5);
		conv_param->add_stride(1);
		conv_param->add_pad(2);

		NeighborDistLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}