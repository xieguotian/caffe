#ifndef CAFFE_CONV_NORM_LAYER_HPP_
#define CAFFE_CONV_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"
namespace caffe {

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
}
#endif