#ifndef CAFFE_SMOOTH_THRESHOLD_LAYER_HPP_
#define CAFFE_SMOOTH_THRESHOLD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

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

		virtual inline const char* type() const { return "SmoothThreshold"; }
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
}
#endif