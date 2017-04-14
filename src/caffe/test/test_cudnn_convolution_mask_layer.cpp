#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_mask_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv_2(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  const bool has_depth = (out->num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  } else {
    kernel_h = kernel_w = conv_param->kernel_size(0);
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  } else {
    pad_h = pad_w = conv_param->pad_size() ? conv_param->pad(0) : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  } else {
    stride_h = stride_w = conv_param->stride_size() ? conv_param->stride(0) : 1;
  }
  int dilation_h, dilation_w;
  dilation_h = dilation_w = conv_param->dilation_size() ?
                            conv_param->dilation(0) : 1;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
            for (int y = 0; y < out->shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in->shape(2) : 1)
                          && in_y >= 0 && in_y < in->shape(2 + has_depth)
                          && in_x >= 0 && in_x < in->shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->shape(0); n++) {
      for (int o = 0; o < out->shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape(2) : 1); z++) {
          for (int y = 0; y < out->shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->offset(out_offset)] += bias_data[o];
            }
          }
        }
      }
    }
  }

  int width = out->width();
  int height = out->height();
  int channels = out->channels()/9;
  vector<int> top_shape = out->shape();
  top_shape[1] /= 9;
  Blob<Dtype> tmp_out;
  tmp_out.Reshape(top_shape);
  for (int n = 0; n < out->num(); ++n)
  {
	  for (int ch = 0; ch < out->channels() / 9; ++ch)
	  {
		  for (int h = 0; h < out->height(); ++h)
		  {
			  for (int w = 0; w < out->width(); ++w)
			  {
				  Dtype max_val = -FLT_MAX;
				  Dtype d[9];
				  for (int i = 0; i < 3; i++)
				  {
					  for (int j = 0; j < 3; j++)
					  {
						  int g_idx = i * 3 + j;
						  int tmp_w_idx = j - 1 + w;
						  int tmp_h_idx = i - 1 + h;
						  if (tmp_w_idx < 0 || tmp_w_idx >= width || tmp_h_idx < 0 || tmp_h_idx >= height)
							  d[g_idx] = 0;
						  else
							  d[g_idx] = out->cpu_data()[(((n * 9 + g_idx)*channels + ch)*height + tmp_h_idx)*width + tmp_w_idx];
					  }
				  }

				  Dtype val[6];
				  val[0] = d[4];
				  val[1] = d[3] + d[4] + d[5];
				  val[2] = d[1] + d[4] + d[7];
				  val[3] = d[0] + d[4] + d[8];
				  val[4] = d[2] + d[4] + d[6];
				  val[5] = d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] + d[8];

				  for (int i = 0; i < 6; i++)
				  {
					  if (max_val < val[i])
					  {
						  max_val = val[i];
					  }
				  }

				  tmp_out.mutable_cpu_data()[((n*channels + ch)*height + h)*width + w] = max_val;
				  //tmp_out.mutable_cpu_data()[((n*channels + ch)*height + h)*width + w] = val[5];
			  }
		  }
	  }
  }
  caffe_copy(tmp_out.count(), tmp_out.cpu_data(), out->mutable_cpu_data());
}

template void caffe_conv_2(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv_2(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);


#ifdef USE_CUDNN

template <typename Dtype>
class CuDNNConvolutionMaskLayerTest : public GPUDeviceTest<Dtype> {
 protected:
	 CuDNNConvolutionMaskLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CuDNNConvolutionMaskLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
	vector<int> top_shape = top->shape();
	top_shape[1] *= 9;
    //this->ref_blob_top_->ReshapeLike(*top);
	this->ref_blob_top_->Reshape(top_shape);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CuDNNConvolutionMaskLayerTest, TestDtypes);

TYPED_TEST(CuDNNConvolutionMaskLayerTest, TestSetupCuDNN) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4*9);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<TypeParam> > layer(
	  new CuDNNConvolutionMaskLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 6);
  EXPECT_EQ(this->blob_top_2_->width(), 4);
  // setting group should not change the shape
  convolution_param->set_num_output(3*9);
  convolution_param->set_group(3);
  layer.reset(new CuDNNConvolutionMaskLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 6);
  EXPECT_EQ(this->blob_top_2_->width(), 4);
}

TYPED_TEST(CuDNNConvolutionMaskLayerTest, TestSimpleConvolutionCuDNN) {
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(4*9);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
	  new CuDNNConvolutionMaskLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_conv_2(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_conv_2(this->blob_bottom_2_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

//TYPED_TEST(CuDNNConvolutionLayerTest, TestSimpleConvolutionGroupCuDNN) {
//  LayerParameter layer_param;
//  ConvolutionParameter* convolution_param =
//      layer_param.mutable_convolution_param();
//  convolution_param->add_kernel_size(3);
//  convolution_param->add_stride(2);
//  convolution_param->set_num_output(3);
//  convolution_param->set_group(3);
//  convolution_param->mutable_weight_filler()->set_type("gaussian");
//  convolution_param->mutable_bias_filler()->set_type("constant");
//  convolution_param->mutable_bias_filler()->set_value(0.1);
//  shared_ptr<Layer<TypeParam> > layer(
//      new CuDNNConvolutionLayer<TypeParam>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//  // Check against reference convolution.
//  const TypeParam* top_data;
//  const TypeParam* ref_top_data;
//  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
//      this->MakeReferenceTop(this->blob_top_));
//  top_data = this->blob_top_->cpu_data();
//  ref_top_data = this->ref_blob_top_->cpu_data();
//  for (int i = 0; i < this->blob_top_->count(); ++i) {
//    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//  }
//}

//TYPED_TEST(CuDNNConvolutionLayerTest, TestSobelConvolutionCuDNN) {
//  // Test separable convolution by computing the Sobel operator
//  // as a single filter then comparing the result
//  // as the convolution of two rectangular filters.
//
//  // Fill bottoms with identical Gaussian noise.
//  shared_ptr<GaussianFiller<TypeParam> > filler;
//  FillerParameter filler_param;
//  filler_param.set_value(1.);
//  filler.reset(new GaussianFiller<TypeParam>(filler_param));
//  filler->Fill(this->blob_bottom_);
//  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
//  // Compute Sobel G_x operator as 3 x 3 convolution.
//  LayerParameter layer_param;
//  ConvolutionParameter* convolution_param =
//      layer_param.mutable_convolution_param();
//  convolution_param->add_kernel_size(3);
//  convolution_param->add_stride(2);
//  convolution_param->set_num_output(1);
//  convolution_param->set_bias_term(false);
//  shared_ptr<Layer<TypeParam> > layer(
//      new CuDNNConvolutionLayer<TypeParam>(layer_param));
//  layer->blobs().resize(1);
//  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 3));
//  TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();
//  for (int c = 0; c < 3; ++c) {
//    int i = c * 9;  // 3 x 3 filter
//    weights[i +  0] = -1;
//    weights[i +  1] =  0;
//    weights[i +  2] =  1;
//    weights[i +  3] = -2;
//    weights[i +  4] =  0;
//    weights[i +  5] =  2;
//    weights[i +  6] = -1;
//    weights[i +  7] =  0;
//    weights[i +  8] =  1;
//  }
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
//  // (1) the [1 2 1] column filter
//  vector<Blob<TypeParam>*> sep_blob_bottom_vec;
//  vector<Blob<TypeParam>*> sep_blob_top_vec;
//  shared_ptr<Blob<TypeParam> > blob_sep(new Blob<TypeParam>());
//  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
//  sep_blob_top_vec.push_back(this->blob_top_2_);
//  convolution_param->clear_kernel_size();
//  convolution_param->clear_stride();
//  convolution_param->set_kernel_h(3);
//  convolution_param->set_kernel_w(1);
//  convolution_param->set_stride_h(2);
//  convolution_param->set_stride_w(1);
//  convolution_param->set_num_output(1);
//  convolution_param->set_bias_term(false);
//  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
//  layer->blobs().resize(1);
//  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 1));
//  TypeParam* weights_1 = layer->blobs()[0]->mutable_cpu_data();
//  for (int c = 0; c < 3; ++c) {
//    int i = c * 3;  // 3 x 1 filter
//    weights_1[i +  0] = 1;
//    weights_1[i +  1] = 2;
//    weights_1[i +  2] = 1;
//  }
//  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
//  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
//  // (2) the [-1 0 1] row filter
//  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
//  sep_blob_bottom_vec.clear();
//  sep_blob_bottom_vec.push_back(blob_sep.get());
//  convolution_param->set_kernel_h(1);
//  convolution_param->set_kernel_w(3);
//  convolution_param->set_stride_h(1);
//  convolution_param->set_stride_w(2);
//  convolution_param->set_num_output(1);
//  convolution_param->set_bias_term(false);
//  layer.reset(new CuDNNConvolutionLayer<TypeParam>(layer_param));
//  layer->blobs().resize(1);
//  layer->blobs()[0].reset(new Blob<TypeParam>(1, 1, 1, 3));
//  TypeParam* weights_2 = layer->blobs()[0]->mutable_cpu_data();
//  weights_2[0] = -1;
//  weights_2[1] =  0;
//  weights_2[2] =  1;
//  layer->SetUp(sep_blob_bottom_vec, sep_blob_top_vec);
//  layer->Forward(sep_blob_bottom_vec, sep_blob_top_vec);
//  // Test equivalence of full and separable filters.
//  const TypeParam* top_data = this->blob_top_->cpu_data();
//  const TypeParam* sep_top_data = this->blob_top_2_->cpu_data();
//  for (int i = 0; i < this->blob_top_->count(); ++i) {
//    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
//  }
//}

TYPED_TEST(CuDNNConvolutionMaskLayerTest, TestGradientCuDNN) {
  LayerParameter layer_param; 
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  //this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  //this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(1);
  convolution_param->add_stride(1);
  convolution_param->set_num_output(2*9);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNConvolutionMaskLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-3, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CuDNNConvolutionMaskLayerTest, TestSpatialReLUGradientCuDNN) {
	LayerParameter layer_param;
	ConvolutionParameter* convolution_param =
		layer_param.mutable_convolution_param();
	//this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
	//this->blob_top_vec_.push_back(this->blob_top_2_);
	convolution_param->add_kernel_size(1);
	convolution_param->add_stride(1);
	convolution_param->set_num_output(2 * 9);
	convolution_param->mutable_weight_filler()->set_type("gaussian");
	convolution_param->mutable_bias_filler()->set_type("gaussian");
	layer_param.mutable_conv_mask_param()->set_use_spatial(true);
	CuDNNConvolutionMaskLayer<TypeParam> layer(layer_param);
	GradientChecker<TypeParam> checker(1e-3, 1e-2);
	checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
		this->blob_top_vec_);
}
//TYPED_TEST(CuDNNConvolutionLayerTest, TestGradientGroupCuDNN) {
//  LayerParameter layer_param;
//  ConvolutionParameter* convolution_param =
//      layer_param.mutable_convolution_param();
//  convolution_param->add_kernel_size(3);
//  convolution_param->add_stride(2);
//  convolution_param->set_num_output(3);
//  convolution_param->set_group(3);
//  convolution_param->mutable_weight_filler()->set_type("gaussian");
//  convolution_param->mutable_bias_filler()->set_type("gaussian");
//  CuDNNConvolutionLayer<TypeParam> layer(layer_param);
//  GradientChecker<TypeParam> checker(1e-2, 1e-3);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}

#endif

}  // namespace caffe
