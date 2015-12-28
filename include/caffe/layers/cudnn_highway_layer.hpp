#ifndef CAFFE_CUDNN_HIGHWAY_LAYER_HPP_
#define CAFFE_CUDNN_HIGHWAY_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#ifdef USE_CUDNN

	template <typename Dtype>
	class CuDNNHighwayLayer : public Layer<Dtype> {
	public:
		explicit CuDNNHighwayLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual ~CuDNNHighwayLayer();

		virtual inline const char* type() const { return "CuDNNHighway"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		// Compute height_out_ and width_out_ from other parameters.
		virtual void compute_output_shape();

		int kernel_h_, kernel_w_;
		int trans_kernel_h_, trans_kernel_w_;
		int stride_h_, stride_w_;
		int num_;
		int channels_;
		int pad_h_, pad_w_;
		int trans_pad_h_, trans_pad_w_;
		int height_, width_;
		int group_;
		int num_output_;
		int height_out_, width_out_;
		bool bias_term_;
		bool is_1x1_;

		int conv_out_channels_;
		int conv_in_channels_;
		int conv_out_spatial_dim_;
		int conv_in_height_;
		int conv_in_width_;
		int kernel_dim_;
		int trans_kernel_dim_;
		int output_offset_;

		Blob<Dtype> bias_multiplier_;

		vector<shared_ptr<Blob<Dtype> > > cell_states;
		vector<shared_ptr<Blob<Dtype> > > transform_gate_states;

		bool handles_setup_;
		cudnnHandle_t* handle_;
		cudaStream_t*  stream_;
		vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
		cudnnTensorDescriptor_t    bias_desc_;
		cudnnFilterDescriptor_t      filter_desc_;
		cudnnFilterDescriptor_t      trans_filter_desc_;
		vector<cudnnConvolutionDescriptor_t> conv_descs_;
		vector<cudnnConvolutionDescriptor_t> trans_conv_descs_;
		int bottom_offset_, top_offset_, weight_offset_,
			trans_weight_offset_, bias_offset_;
		size_t workspaceSizeInBytes;
		void *workspace;
	};

#endif
}
#endif