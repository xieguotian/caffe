#ifndef CAFFE_SELECT_REPLACE_LAYER_HPP_
#define CAFFE_SELECT_REPLACE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class SelectReplaceLayer : public Layer<Dtype> {
	public:
		explicit SelectReplaceLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "SelectReplace"; }
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
}
#endif