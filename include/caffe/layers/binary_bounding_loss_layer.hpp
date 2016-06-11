#ifndef CAFFE_BINARY_BOUNDING_LOSS_LAYER_HPP_
#define CAFFE_BINARY_BOUNDING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe{
	template <typename Dtype>
	class BinaryBoundingLossLayer : public LossLayer<Dtype> {
	public:
		/**
		* @param param provides LossParameter loss_param, with options:
		*  - ignore_label (optional)
		*    Specify a label value that should be ignored when computing the loss.
		*  - normalize (optional, default true)
		*    If true, the loss is normalized by the number of (nonignored) labels
		*    present; otherwise the loss is simply summed over spatial locations.
		*/
		explicit BinaryBoundingLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "BinaryBoundingLoss"; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }
		virtual inline int ExactNumBottomTopBlobs() const{ return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top)
		{
			NOT_IMPLEMENTED;
		}
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
		{
			NOT_IMPLEMENTED;
		}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		bool update_centroid = false;
		Dtype alpha;
		Dtype beta;
		Dtype ratio;
		Blob<Dtype> cache_tmp_;
		Blob<Dtype> square_cache_tmp_;
		Blob<Dtype> scalar_cache_;
		Blob<Dtype> ones_column;
		int dustbin_label;
		Dtype threshold;
		bool not_initialed;
		Blob<Dtype> ones_;
	};
}

#endif
