#ifndef CAFFE_NON_LOCAL_LAYER_HPP_
#define CAFFE_NON_LOCAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/smooth_threshold_layer.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

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

		virtual inline const char* type() const { return "NonLocal"; }
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
}
#endif