#ifndef CAFFE_CLUSTER_CENTROID_DIST_LAYER_HPP_
#define CAFFE_CLUSTER_CENTROID_DIST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	template <typename Dtype>
	class ClusterCentroidDistLayer : public Layer<Dtype> {
	public:
		explicit ClusterCentroidDistLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ClusterCentroidDist"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		//virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		/// @brief Not implemented (non-differentiable function)
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			NOT_IMPLEMENTED;
		}
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int centroid_dim_;
		int num_cluster_;
		Blob<Dtype> cache_feat_;
		Blob<Dtype> cache_cluster_;
		Blob<Dtype> ones_;
		Blob<Dtype> column_;
		Blob<Dtype> square_feat_;
		Blob<Dtype> square_cluster_;
		float scale;

		bool initialized_;
		int init_count_;
		Blob<Dtype> top_cache_;
		Blob<Dtype> bottom_cache_;
		float T;
		bool use_T_;
		//Blob<Dtype> temp_diff_;
		bool is_sample_base_cls;
		//bool is_spatial_;
		//Blob<Dtype> transposed_bottom_;
		//Blob<Dtype> transposed_top_;
		int num_samp_;
	};
}

#endif
