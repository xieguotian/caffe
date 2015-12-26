#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe\layers\split_layer.hpp"
#include "caffe\layers\eltwise_layer.hpp"
#include "caffe\layers\conv_layer.hpp"

namespace caffe {
/* UnPoolingLayer
*/
template <typename Dtype>
class UnPoolingLayer : public Layer<Dtype> {
public:
	explicit UnPoolingLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const {
		return "UnPooling";
	}


	virtual inline int ExactNumBottomBlobs() const 
	{
		return (this->layer_param_.pooling_param().pool() ==
			PoolingParameter_PoolMethod_MAX ? 3 : 1);
	}
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int max_top_blobs_;
	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int pad_h_, pad_w_;
	int channels_;
	int height_;
	int width_;
	int pooled_height_;
	int pooled_width_;
	bool global_pooling_;
};

template <typename Dtype>
class KSparseLayer : public Layer<Dtype>
{
public:
	explicit KSparseLayer(const LayerParameter& param)
	: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "KSparse"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 2; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int sparse_k;
	Dtype decay;
	Blob<int> rank_idx_;
	Blob<Dtype> rank_val_;
};

template <typename Dtype>
class DeconvNormLayer : public Layer<Dtype>
{
public:
	explicit DeconvNormLayer(const LayerParameter& param)
	: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "DeconvNorm"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


	bool bias_term_;
	bool average_train;

	Blob<Dtype> constant1;
	Blob<Dtype> coef_norm;
	Blob<Dtype> deconv_val;
	shared_ptr<Blob<Dtype>> alphas;

	Blob<Dtype> deconv1_top_cache;
	Blob<Dtype> alpha_cache;
	Blob<Dtype> alpha_cache2;
	shared_ptr<Blob<Dtype>> weights_alphas;

	vector<Blob<Dtype>*> deconv1_bottom_vec;
	vector<Blob<Dtype>*> deconv1_top_vec;
	vector<Blob<Dtype>*> deconv2_top_vec;
	shared_ptr<DeconvolutionLayer<Dtype>> deconv1_layer;
	shared_ptr<DeconvolutionLayer<Dtype>> deconv2_layer;
	vector<Blob<Dtype>*> exp_top_vec;
	vector<Blob<Dtype>*> exp_bottom_vec;
	shared_ptr<ExpLayer<Dtype>> exp_layer;

	Blob<Dtype> bias_multiplier;
};

template <typename Dtype>
class ConvNormLayer : public Layer<Dtype>
{
public:
	explicit ConvNormLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ConvNorm"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	shared_ptr<ConvolutionLayer<Dtype>> conv_layer;
	shared_ptr<ConvolutionLayer<Dtype>> norm_layer;
	vector<Blob<Dtype>*> conv_top_vec;
	//vector<Blob<Dtype>*> conv_bottom_vec;
	vector<Blob<Dtype>*> norm_top_vec;
	vector<Blob<Dtype>*> norm_bottom_vec;

	Blob<Dtype> conv_top;
	//Blob<Dtype> conv_bottom;
	Blob<Dtype> norm_top;
	Blob<Dtype> norm_bottom;
};


/*
f(x,c) = 1/(1+alpha*exp((beta*(-x+c))
c is threshold
x is data
*/
template <typename Dtype>
class SmoothThresholdLayer : public Layer<Dtype>
{
public:
	explicit SmoothThresholdLayer(const LayerParameter& param)
		: Layer<Dtype>(param), diff() {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SmoothThresholdLayer"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Dtype alpha;
	Dtype beta;
	Blob<Dtype> diff;
};


template <typename Dtype>
class EuclideanLayer : public Layer<Dtype>
{
public:
	explicit EuclideanLayer(const LayerParameter& param)
		: Layer<Dtype>(param), diff() {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "EuclideanLayer"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> diff;
	Blob<Dtype> square;
	Blob<Dtype> const_one;
};

template <typename Dtype>
class NormalizeLayer : public Layer<Dtype> {
public:
	explicit NormalizeLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Normalize"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> norm_cache_;
	Blob<Dtype> ones_;
};

template <typename Dtype>
class NonLocalLayer : public Layer<Dtype>
{
public:
	explicit NonLocalLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "NonLocalLayer"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 2; }

	virtual inline int MaxTopBlobs() const { return 3; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int num_;
	int channels_;
	int pad_h_, pad_w_;
	int height_, width_;
	int height_out_, width_out_;
	int num_output_;
	bool is_1x1_;

	Blob<Dtype> smooth_top;
	Blob<Dtype> euclidean_top;
	Blob<Dtype> eltwise_top;
	Blob<Dtype> normalize_top;

	Blob<Dtype> split_0_top_0;
	Blob<Dtype> split_0_top_1;
	Blob<Dtype> split_1_top_0;
	Blob<Dtype> split_1_top_1;
	Blob<Dtype> split_2_top_0;
	Blob<Dtype> split_3_top_0;
	Blob<Dtype> split_3_top_1;

	Blob<Dtype> img2col_0_top;
	Blob<Dtype> img2col_1_top;
	Blob<Dtype> mask_top;

	Blob<Dtype> euclidean_bottom_0;
	Blob<Dtype> euclidean_bottom_1;
	Blob<Dtype> split_2_bottom;
	Blob<Dtype> normalize_bottom;

	vector<Blob<Dtype>*> smooth_top_vec;
	vector<Blob<Dtype>*> smooth_bottom_vec;
	vector<Blob<Dtype>*> euclidean_top_vec;
	vector<Blob<Dtype>*> euclidean_bottom_vec;
	vector<Blob<Dtype>*> eltwise_top_vec;
	vector<Blob<Dtype>*> eltwise_bottom_vec;
	vector<Blob<Dtype>*> normalize_bottom_vec;
	vector<Blob<Dtype>*> normalize_top_vec;

	vector<Blob<Dtype>*> split_0_top_vec;
	vector<Blob<Dtype>*> split_1_top_vec;
	vector<Blob<Dtype>*> split_1_bottom_vec;
	vector<Blob<Dtype>*> split_2_top_vec;
	vector<Blob<Dtype>*> split_2_bottom_vec;
	vector<Blob<Dtype>*> split_3_top_vec;
	vector<Blob<Dtype>*> split_3_bottom_vec;

	shared_ptr<SmoothThresholdLayer<Dtype>> smooth_threshold_layer;
	shared_ptr<EuclideanLayer<Dtype>> euclidean_layer;
	shared_ptr<EltwiseLayer<Dtype>> eltwise_layer;

	shared_ptr<SplitLayer<Dtype>> split_layer_0;
	shared_ptr<SplitLayer<Dtype>> split_layer_1;
	shared_ptr<SplitLayer<Dtype>> split_layer_2;
	shared_ptr<SplitLayer<Dtype>> split_layer_3;
	shared_ptr<NormalizeLayer<Dtype>> normalize_layer;
};

//f(x,sita)=sign(xi)(xi-sita_i)+
template <typename Dtype>
class ShrinkageLayer : public Layer<Dtype>
{
public:
	explicit ShrinkageLayer(const LayerParameter& param)
		: Layer<Dtype>(param){}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "shrinkage"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> sign_x;
	Blob<Dtype> ones;
};

template <typename Dtype>
class Col2imgMaskLayer : public Layer<Dtype> {
public:
	explicit Col2imgMaskLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "Col2imgMask"; }
	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int channels_;
	int height_, width_;
	int pad_h_, pad_w_;
	int height_out_, width_out_;
	int channels_out_;

	Blob<Dtype> mask_out_;
	Blob<Dtype> mask_in_;
	Blob<Dtype> eltwise_top;
	Blob<Dtype> split_top_0;
	Blob<Dtype> split_top_1;
	Blob<Dtype> data_out_;

	vector<Blob<Dtype>*> eltwise_bottom_vec;
	vector<Blob<Dtype>*> eltwise_top_vec;
	vector<Blob<Dtype>*> split_bottom_vec;
	vector<Blob<Dtype>*> split_top_vec;

	shared_ptr<EltwiseLayer<Dtype>> eltwise_layer;
	shared_ptr<SplitLayer<Dtype>> split_layer;
};

template <typename Dtype>
class NonLocal2Layer : public Layer<Dtype>
{
public:
	explicit NonLocal2Layer(const LayerParameter& param)
		: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "NonLocal2Layer"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int num_;
	int channels_;
	int pad_h_, pad_w_;
	int height_, width_;
	int height_out_, width_out_;
	int num_output_;
	bool is_1x1_;

	Blob<Dtype> exp_top;
	Blob<Dtype> euclidean_top;
	Blob<Dtype> normalize_top;

	Blob<Dtype> split_0_top_0;
	Blob<Dtype> split_0_top_1;

	Blob<Dtype> img2col_0_top;
	Blob<Dtype> img2col_1_top;

	Blob<Dtype> euclidean_bottom_0;
	Blob<Dtype> euclidean_bottom_1;
	Blob<Dtype> normalize_bottom;

	vector<Blob<Dtype>*> exp_top_vec;
	vector<Blob<Dtype>*> exp_bottom_vec;
	vector<Blob<Dtype>*> euclidean_top_vec;
	vector<Blob<Dtype>*> euclidean_bottom_vec;
	vector<Blob<Dtype>*> normalize_bottom_vec;
	vector<Blob<Dtype>*> normalize_top_vec;

	vector<Blob<Dtype>*> split_0_top_vec;

	shared_ptr<ExpLayer<Dtype>> exp_layer;
	shared_ptr<EuclideanLayer<Dtype>> euclidean_layer;

	shared_ptr<SplitLayer<Dtype>> split_layer_0;
	shared_ptr<NormalizeLayer<Dtype>> normalize_layer;
};

template <typename Dtype>
class AmplitudeLayer : public Layer<Dtype>
{
public:
	explicit AmplitudeLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "AmplitudeLayer"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }

	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> square;
	Blob<Dtype> const_one;
};

template <typename Dtype>
class NeighborDistLayer : public Layer<Dtype> {
public:
	explicit NeighborDistLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "NeighborDistLayer"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int num_;
	int channels_;
	int pad_h_, pad_w_;
	int height_, width_;
	int height_out_, width_out_;
	int num_output_;

};

template <typename Dtype>
class SelectSortedLayer : public Layer<Dtype> {
public:
	explicit SelectSortedLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SelectSortedLayer"; }
	virtual inline int MinBottomBlobs() const { return 1; }
	//virtual inline int MaxBottomBlobs() const { return 2; }
	virtual inline int MinNumTopBlobs() const { return 1; }
	//virtual inline int MaxNumTopBlobs() const { return 2; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int num_;
	int channels_;
	int pad_h_, pad_w_;
	int height_, width_;
	int height_out_, width_out_;
	int num_output_;
	int top_N_;
	shared_ptr<NeighborDistLayer<Dtype>> nei_dist_layer;
	vector<Blob<Dtype>*> nei_dist_top_vec;
	vector<Blob<Dtype>*> nei_dist_bottom_vec;

	Blob<Dtype> dist;
	Blob<Dtype> key;
	Blob<Dtype> index;
};

template <typename Dtype>
class SelectReplaceLayer : public Layer<Dtype> {
public:
	explicit SelectReplaceLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SelectReplaceLayer"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int MinNumTopBlobs() const { return 1; }
	virtual inline int MaxNumTopBlobs() const { return 2; }
	//virtual inline int ExactNumTopBlobs() const { return 1; }


protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int kernel_h_, kernel_w_;
	int stride_h_, stride_w_;
	int num_;
	int channels_;
	int pad_h_, pad_w_;
	int height_, width_;
	int height_out_, width_out_;
	int num_output_;
	int top_N_;

	Blob<Dtype> count_coef_;
	Blob<Dtype> idx_trans_cache_;

};

}  // namespace caffe

#endif  // CAFFE_VISION_LAYERS_HPP_
