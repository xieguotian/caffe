#ifndef CAFFE_ARGMAXMIN_LAYER_HPP_
#define CAFFE_ARGMAXMIN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class ArgMaxMinLayer : public Layer<Dtype> {
	public:
		/**
		* @param param provides ArgMaxParameter argmax_param,
		*     with ArgMaxLayer options:
		*/
		explicit ArgMaxMinLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ArgMaxMin"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		//virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline int  MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:

		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
		{
			if (!propagate_down[0])
				return;
			NOT_IMPLEMENTED;
		}
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top)
		{
			Forward_gpu(bottom, top);
			//NOT_IMPLEMENTED;
		}
		/// @brief Not implemented (non-differentiable function)
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			if (!propagate_down[0])
				return;
			NOT_IMPLEMENTED;
		}

		bool is_max_;

	};
}
#endif