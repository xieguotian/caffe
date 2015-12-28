#ifndef CAFFE_COL2IMG_MASK_LAYER_HPP_
#define CAFFE_COL2IMG_MASK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"
namespace caffe {

	template <typename Dtype>
	class Col2imgMaskLayer : public Layer<Dtype> {
	public:
		explicit Col2imgMaskLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Col2imgMask"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MaxBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

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
		int channels_;
		int height_, width_;
		int pad_h_, pad_w_;
		int height_out_, width_out_;
		int channels_out_;

		Blob<Dtype> mask_out_;
		Blob<Dtype> mask_in_;
		Blob<Dtype> eltwise_top;
		Blob<Dtype> split_top_0;
		Blob<Dtype> split_top_1;
		Blob<Dtype> data_out_;

		vector<Blob<Dtype>*> eltwise_bottom_vec;
		vector<Blob<Dtype>*> eltwise_top_vec;
		vector<Blob<Dtype>*> split_bottom_vec;
		vector<Blob<Dtype>*> split_top_vec;

		shared_ptr<EltwiseLayer<Dtype>> eltwise_layer;
		shared_ptr<SplitLayer<Dtype>> split_layer;
	};
}
#endif