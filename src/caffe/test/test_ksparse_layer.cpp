#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/k_sparse_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class KSparseLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		KSparseLayerTest()
			: blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()),
			t1(new Blob<Dtype>()),
			t3(new Blob<Dtype>()),
			t5(new Blob<Dtype>()){}

		virtual void SetUp() {
			Caffe::set_random_seed(1701);
			blob_bottom_->Reshape(2, 5, 2, 2);
			Dtype* bottom_data = blob_bottom_->mutable_cpu_data();
			Dtype data[] = { 7, 4, 29, 22, 20,
				26, 10, 21, 36, 39,
				12, 11, 24, 37, 15,
				8, 31, 34, 27, 5,
				0, 30, 14, 16, 1,
				6, 13, 3, 23, 28,
				9, 2, 32, 38, 19,
				17, 25, 35, 18, 33 };
			caffe_copy(blob_bottom_->count(), data, bottom_data);

			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);

			Dtype t1_data[] = { 0, 0, 29, 22, 0,
				0, 0, 0, 36, 39,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 32, 38, 19,
				0, 0, 0, 0, 33 };
			Dtype t3_data[] = { 0, 0, 29, 22, 0,
				0, 0, 21, 36, 39,
				0, 11, 24, 37, 15,
				0, 31, 34, 27, 0,
				0, 30, 14, 16, 0,
				0, 0, 0, 23, 0,
				0, 0, 32, 38, 19,
				17, 25, 35, 18, 33 };
			t1->ReshapeLike(*blob_bottom_);
			t3->ReshapeLike(*blob_bottom_);
			t5->ReshapeLike(*blob_bottom_);
			caffe_copy(t1->count(), t1_data, t1->mutable_cpu_data());
			caffe_copy(t3->count(), t3_data, t3->mutable_cpu_data());
			caffe_copy(t5->count(), data, t5->mutable_cpu_data());
		}

		virtual ~KSparseLayerTest() {
			delete blob_bottom_;
			delete blob_top_;
		}

		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;

		Blob<Dtype>* t1;
		Blob<Dtype>* t3;
		Blob<Dtype>* t5;
	};

	TYPED_TEST_CASE(KSparseLayerTest, TestDtypesAndDevices);

	TYPED_TEST(KSparseLayerTest, TestSetup)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(2);
		ksparse_param->set_sparse_type(KSparseParameter_SparseMethod_CHANNEL);
		ksparse_param->set_type(KSparseParameter_SparseType_KSPARSE);
		KSparseLayer<Dtype> layer(layer_param);
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
	}

	TYPED_TEST(KSparseLayerTest, TestTopOne)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(1);
		ksparse_param->set_sparse_type(KSparseParameter_SparseMethod_CHANNEL);
		ksparse_param->set_type(KSparseParameter_SparseType_KSPARSE);
		KSparseLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		for (int i = 0; i < this->blob_top_->count(); i++)
		{
			EXPECT_EQ(this->blob_top_->cpu_data()[i], this->t1->cpu_data()[i])
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< this->t1->cpu_data()[i];
		}
	}

	TYPED_TEST(KSparseLayerTest, TestTopThree)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(3);
		ksparse_param->set_sparse_type(KSparseParameter_SparseMethod_CHANNEL);
		ksparse_param->set_type(KSparseParameter_SparseType_KSPARSE);
		KSparseLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		for (int i = 0; i < this->blob_top_->count(); i++)
		{
			EXPECT_EQ(this->blob_top_->cpu_data()[i], this->t3->cpu_data()[i])
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< this->t3->cpu_data()[i];
		}
	}

	TYPED_TEST(KSparseLayerTest, TestTopFive)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(5);
		ksparse_param->set_sparse_type(KSparseParameter_SparseMethod_CHANNEL);
		ksparse_param->set_type(KSparseParameter_SparseType_KSPARSE);
		KSparseLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

		for (int i = 0; i < this->blob_top_->count(); i++)
		{
			EXPECT_EQ(this->blob_top_->cpu_data()[i], this->t5->cpu_data()[i])
				<< "(top_data,gt_data)=" << this->blob_top_->cpu_data()[i] << ","
				<< this->t5->cpu_data()[i];
		}
	}

	TYPED_TEST(KSparseLayerTest, TestGradient)
	{
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		KSparseParameter* ksparse_param = layer_param.mutable_k_sparse_param();
		ksparse_param->set_sparse_k(3);
		ksparse_param->set_sparse_type(KSparseParameter_SparseMethod_CHANNEL);
		ksparse_param->set_type(KSparseParameter_SparseType_KSPARSE);
		KSparseLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);

	}
}