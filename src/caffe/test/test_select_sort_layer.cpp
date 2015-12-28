#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/select_sorted_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
	template <typename TypeParam>
	class SelectSortedLayerTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		SelectSortedLayerTest() :blob_bottom_0(new Blob<Dtype>(2, 2, 2, 2)),
			blob_bottom_1(new Blob<Dtype>(2, 2, 2, 2)),
			blob_top_0(new Blob<Dtype>()), blob_top_1(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_0);
			blob_bottom_vec_.push_back(blob_bottom_1);
			blob_top_vec_.push_back(blob_top_0);
			blob_top_vec_.push_back(blob_top_1);
			
		}

		virtual ~SelectSortedLayerTest() { delete blob_bottom_0;
		delete blob_bottom_1; delete blob_top_0; delete blob_top_1;
		}


		Blob<Dtype>* const blob_bottom_0;
		Blob<Dtype>* const blob_bottom_1;
		Blob<Dtype>* const blob_top_0;
		Blob<Dtype>* const blob_top_1;

		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(SelectSortedLayerTest, TestDtypesAndDevices);

	TYPED_TEST(SelectSortedLayerTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input_1[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };
		Dtype input_2[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };
		//Dtype output_1[] = { 0., 1., 2., 3., 1., 2., 1., 1., 0., 1., 2., 3., 1., 3., 3., 2. };
		Dtype output_1[] = { 4., 4., 4., 4., 5., 6., 2., 1., 4., 4., 4., 4., 5., 7., 5., 3. };
		Dtype output_2[] = { 15, 13, 12, 0, 13, 12, 13, 13, 5, 8, 11, 4, 8, 11, 8, 8, 14,
			6, 7, 3, 6, 3, 3, 7, 10, 9, 1, 2, 9, 2, 2, 1 };

		caffe_copy(blob_bottom_0->count(), input_1, blob_bottom_0->mutable_cpu_data());
		caffe_copy(blob_bottom_1->count(), input_2, blob_bottom_1->mutable_cpu_data());

		LayerParameter layer_param;

		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(3);
		conv_param->add_stride(1);
		conv_param->add_pad(1);
		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(2);

		SelectSortedLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(this->blob_top_0->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_0->num() << ","
			<< this->blob_bottom_0->num();
		EXPECT_EQ(this->blob_top_0->channels(),2)
			<< "(top_channels,bottom_channels)=" << this->blob_top_0->channels() << ","
			<< this->blob_bottom_0->channels();
		EXPECT_EQ(this->blob_top_0->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_0->height() << ","
			<< this->blob_bottom_0->height();
		EXPECT_EQ(this->blob_top_0->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_0->width() << ","
			<< this->blob_bottom_0->width();

		EXPECT_EQ(this->blob_top_1->num(), 2)
			<< "(top_num,bottom_num)=" << this->blob_top_1->num() << ","
			<< this->blob_bottom_0->num();
		EXPECT_EQ(this->blob_top_1->channels(), 4)
			<< "(top_channels,bottom_channels)=" << this->blob_top_1->channels() << ","
			<< this->blob_bottom_0->channels();
		EXPECT_EQ(this->blob_top_1->height(), 2)
			<< "(top_height,bottom_height)=" << this->blob_top_1->height() << ","
			<< this->blob_bottom_0->height();
		EXPECT_EQ(this->blob_top_1->width(), 2)
			<< "(top_width,bottom_width)=" << this->blob_top_1->width() << ","
			<< this->blob_bottom_0->width();

		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		const Dtype min_precision = 1e-5;
		for (int i = 0; i < blob_top_0->count(); i++)
		{

			EXPECT_NEAR(this->blob_top_0->cpu_data()[i], output_1[i], min_precision)
				<< "(top_data1,gt_data)=" << this->blob_top_0->cpu_data()[i] << ","
				<< output_1[i];

		}
		for (int i = 0; i < blob_top_1->count(); i++)
		{
			EXPECT_NEAR(this->blob_top_1->cpu_data()[i], output_2[i], min_precision)
				<< "(top_data2,gt_data)=" << this->blob_top_1->cpu_data()[i] << ","
				<< output_2[i];
		}
	}

	TYPED_TEST(SelectSortedLayerTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.5);
		GaussianFiller<Dtype> filler(filler_param);
		blob_bottom_0->Reshape(2, 2, 3, 3);
		filler.Fill(blob_bottom_0);
		blob_bottom_1->Reshape(2, 2, 3, 3);
		filler.Fill(blob_bottom_1);
		//Dtype input_1[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };
		//Dtype input_2[] = { 15, 13, 12, 0, 5, 8, 11, 4, 14, 6, 7, 3, 10, 9, 1, 2 };
		//caffe_copy(blob_bottom_0->count(), input_1, blob_bottom_0->mutable_cpu_data());
		//caffe_copy(blob_bottom_1->count(), input_2, blob_bottom_1->mutable_cpu_data());

		LayerParameter layer_param;

		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(3);
		conv_param->add_stride(2);
		conv_param->add_pad(1);

		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(2);
		SelectSortedLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-3, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_,1);
	}
}