#ifndef CAFFE_DECONV_NORM_LAYER_HPP_
#define CAFFE_DECONV_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe\layers\split_layer.hpp"
#include "caffe\layers\eltwise_layer.hpp"
#include "caffe\layers\conv_layer.hpp"
namespace caffe {

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
}
#endif