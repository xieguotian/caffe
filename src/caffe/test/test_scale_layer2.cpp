#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/scale_layer2.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
namespace caffe {

	template <typename TypeParam>
	class Scale2LayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		Scale2LayerTest()
			: blob_bottom_(new Blob<Dtype>(2, 10, 4, 5)),
			blob_top_(new Blob<Dtype>()) {
			Caffe::set_random_seed(1701);
			// fill the values
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~Scale2LayerTest() { delete blob_bottom_;  delete blob_top_; }
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};
	TYPED_TEST_CASE(Scale2LayerTest, TestDtypesAndDevices);
	TYPED_TEST(Scale2LayerTest, TestForward)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->set_num_output(8);
		Scale2Layer<Dtype> layer(layer_param);
		caffe_set(blob_bottom_->count(), (Dtype)1, blob_bottom_->mutable_cpu_data());
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);

		Dtype* param0 = layer.blobs()[0]->mutable_cpu_data();
		caffe_set(layer.blobs()[0]->count(), (Dtype)0, param0);
		param0[1] = 1;
		param0[4] = 1;
		param0[5] = 1;
		param0[9] = 1;
		Dtype* param1 = layer.blobs()[1]->mutable_cpu_data();
		caffe_set(layer.blobs()[1]->count(), (Dtype)-1, param1);
		param1[1] = 3;
		param1[4] = 7;
		param1[5] = 4;
		param1[9] = 5;
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		int ch = blob_top_->channels();
		int height = blob_top_->height();
		int width = blob_top_->width();
		for (int i = 0; i < blob_top_->count(); i++)
		{
			int ch_idx = (i / width / height) % ch;
			if (ch_idx == 3 || ch_idx == 7 || ch_idx == 4 || ch_idx == 5)
				EXPECT_NEAR(blob_top_->mutable_cpu_data()[i], 1, min_precision);
			else
				EXPECT_NEAR(blob_top_->mutable_cpu_data()[i], -1, min_precision);

		}
	}
	TYPED_TEST(Scale2LayerTest, TestBackward) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->set_num_output(8);
		Scale2Layer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		Dtype* param0 = layer.blobs()[0]->mutable_cpu_data();
		caffe_set(layer.blobs()[0]->count(), (Dtype)0, param0);
		param0[1] = 0.2;
		param0[4] = 0.5;
		param0[5] = 0.7;
		param0[9] = 0.3;
		Dtype* param1 = layer.blobs()[1]->mutable_cpu_data();
		caffe_set(layer.blobs()[1]->count(), (Dtype)-1, param1);
		param1[1] = 3;
		param1[4] = 7;
		param1[5] = 4;
		param1[9] = 5;


		GradientChecker<Dtype> checker(1e-2, 1e-2);
		checker.CheckGradientExhaustive2(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}