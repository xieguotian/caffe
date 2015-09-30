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
	class ShrinkageTest : public MultiDeviceTest<TypeParam>
	{
		typedef typename TypeParam::Dtype Dtype;
	protected:
		ShrinkageTest() :blob_bottom_(new Blob<Dtype>(2, 3, 3, 3)),
			blob_top_(new Blob<Dtype>())
		{
			Caffe::set_random_seed(1701);
			FillerParameter filler_param;
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}

		virtual ~ShrinkageTest() { delete blob_bottom_; delete blob_top_; }


		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(ShrinkageTest, TestDtypesAndDevices);

	TYPED_TEST(ShrinkageTest, TestForward_1)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { 0.806748, 2.13815765, -0.61800615, 0.95469029, 0.18660709,
			0.89124599, -1.32282674, -1.53936915, 1.63613597, 0.82837981,
			0.9577062, 2.50513289, -0.65826346, 0.91900327, -0.61297905,
			-0.28486494, 0.44234616, 2.25522211, -0.61370303, -0.63968226,
			-0.33991636, -1.05199342, -2.25413498, -1.26654774, -0.745922,
			-0.06437601, -0.22923511, -0.23479705, -0.48007636, -1.06868447,
			1.07642586, -1.15145357, -0.95259024, -0.11262909, 0.61847481,
			0.39392979, 0.92734781, 0.32648755, 2.92647575, -0.01381278,
			-1.76525753, 0.26058373, 0.72236247, -0.50810132, 0.35037755,
			1.0753405, 1.55860174, -0.08198076, 0.55424671, -0.19967911,
			-0.97463088, 2.58146077, -0.8786181, -3.01485995 };

		Dtype output[] = { 0.306748, 1.63815765, -0.11800615, 0.45469029, 0.,
			0.39124599, -0.82282674, -1.03936915, 1.13613597, 0.32837981,
			0.4577062, 2.00513289, -0.15826346, 0.41900327, -0.11297905,
			0., 0., 1.75522211, -0.11370303, -0.13968226,
			0., -0.55199342, -1.75413498, -0.76654774, -0.245922,
			0., 0., 0., 0., -0.56868447,
			0.57642586, -0.65145357, -0.45259024, 0., 0.11847481,
			0., 0.42734781, 0., 2.42647575, 0.,
			-1.26525753, 0., 0.22236247, -0.00810132, 0.,
			0.5753405, 1.05860174, 0., 0.05424671, 0.,
			-0.47463088, 2.08146077, -0.3786181, -2.51485995 };

		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;
		ShrinkageParameter* shrinkage_param = layer_param.mutable_shrinkage_param();
		shrinkage_param->mutable_threshold_filler()->set_type("constant");
		shrinkage_param->mutable_threshold_filler()->set_value(0.5);

		ShrinkageLayer<Dtype> layer(layer_param);
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

	TYPED_TEST(ShrinkageTest, TestForward_2)
	{
		typedef typename TypeParam::Dtype Dtype;
		Dtype input[] = { -0.33245626, 1.14826835, 0.23703156, 1.42077277, -0.65566234,
			0.91733442, 0.37892521, 0.19659692, 1.73128952, -0.10883285,
			-0.08015601, 1.49597785, -0.22430131, -0.78589101, -0.19171246,
			1.96414529, 1.03288059, -1.45264386, -0.64749458, -0.5176926,
			2.09881133, -0.45996033, -1.07854536, 0.54944002, -0.16418875,
			-0.94213886, 0.91138059, 0.20692938, -0.65483475, -0.74134259,
			0.80947999, 0.97865051, 0.42897463, 0.95084322, 1.02355712,
			-0.74742107, -0.53976948, -0.40380415, -0.76007144, 0.57524043,
			1.0375749, -1.43376636, -0.66025985, -0.75489946, -1.87053951,
			-1.79666323, -0.0304524, 0.06080371, 1.57752797, -0.31666739,
			-1.42572927, 0.12097071, -0.16973385, -0.08045061 };

		Dtype output[] = { 0., 0.64826835, 0., 0.92077277, -0.15566234,
			0.41733442, 0., 0., 1.23128952, 0.,
			0., 0.99597785, 0., -0.28589101, 0.,
			1.46414529, 0.53288059, -0.95264386, -0.14749458, -0.0176926,
			1.59881133, 0., -0.57854536, 0.04944002, 0.,
			-0.44213886, 0.41138059, 0., -0.15483475, -0.24134259,
			0.30947999, 0.47865051, 0., 0.45084322, 0.52355712,
			-0.24742107, -0.03976948, 0., -0.26007144, 0.07524043,
			0.5375749, -0.93376636, -0.16025985, -0.25489946, -1.37053951,
			-1.29666323, 0., 0., 1.07752797, 0.,
			-0.92572927, 0., 0., 0. };

		caffe_copy(blob_bottom_->count(), input, blob_bottom_->mutable_cpu_data());

		LayerParameter layer_param;
		ShrinkageParameter* shrinkage_param = layer_param.mutable_shrinkage_param();
		shrinkage_param->mutable_threshold_filler()->set_type("constant");
		shrinkage_param->mutable_threshold_filler()->set_value(0.5);

		ShrinkageLayer<Dtype> layer(layer_param);
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

	TYPED_TEST(ShrinkageTest, TestBackward)
	{
		typedef typename TypeParam::Dtype Dtype;
		FillerParameter filler_param;
		filler_param.set_std(0.5);
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(blob_bottom_);

		LayerParameter layer_param;
		ShrinkageParameter* shrinkage_param = layer_param.mutable_shrinkage_param();
		shrinkage_param->mutable_threshold_filler()->set_type("constant");
		shrinkage_param->mutable_threshold_filler()->set_value(0.3);

		ShrinkageLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-5, 1e-2, 1701);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
}