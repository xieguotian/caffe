#ifndef CAFFE_NON_LOCAL_2_LAYER_HPP_
#define CAFFE_NON_LOCAL_2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/euclidean_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe/layers/normalize_layer.hpp"
#include "caffe/layers/split_layer.hpp"

namespace caffe {

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

		virtual inline const char* type() const { return "NonLocal2"; }
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
}
#endif
