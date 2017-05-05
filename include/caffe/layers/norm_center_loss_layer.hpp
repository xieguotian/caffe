#ifndef CAFFE_NORM_CENTER_LOSS_LAYER_HPP_
#define CAFFE_NORM_CENTER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	class NormCenterLossLayer : public LossLayer<Dtype> {
	public:
		explicit NormCenterLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "NormCenterLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		void norm_weight_forward_cpu(Dtype* weight,
			Dtype* norm_weight, int n, int d);
		void norm_weight_forward_gpu(Dtype* weight,
			Dtype* norm_weight, int n, int d);
		void norm_weight_backward_cpu(Dtype* top_diff, const Dtype* top_data,
			Dtype* bottom_diff, const Dtype* bottom_data, int n, int d);
		void norm_weight_backward_gpu(Dtype* top_diff, const Dtype* top_data,
			Dtype* bottom_diff, const Dtype* bottom_data, int n, int d);

		int M_;
		int K_;
		int N_;

		Blob<Dtype> distance_;
		Blob<Dtype> center_distance_;
		Blob<Dtype> center_diff_;
		Blob<Dtype> variation_sum_;
		Blob<Dtype> squared_; 
		int iter_;
	};

}  // namespace caffe

#endif  // CAFFE_NORM_CENTER_LOSS_LAYER_HPP_