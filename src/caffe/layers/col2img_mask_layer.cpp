#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void Col2imgMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) 
	{
		ConvolutionParameter conv_param = this->layer_param_.convolution_param();
		CHECK(!conv_param.has_kernel_size() !=
			!(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
			<< "Filter size is kernel_size OR kernel_h and kernel_w; not both";
		CHECK(conv_param.has_kernel_size() ||
			(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
			<< "For non-square filters both kernel_h and kernel_w are required.";
		CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
			&& conv_param.has_pad_w())
			|| (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
			<< "pad is pad OR pad_h and pad_w are required.";
		CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
			&& conv_param.has_stride_w())
			|| (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
			<< "Stride is stride OR stride_h and stride_w are required.";
		if (conv_param.has_kernel_size()) {
			kernel_h_ = kernel_w_ = conv_param.kernel_size();
		}
		else {
			kernel_h_ = conv_param.kernel_h();
			kernel_w_ = conv_param.kernel_w();
		}
		CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
		CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
		if (!conv_param.has_pad_h()) {
			pad_h_ = pad_w_ = conv_param.pad();
		}
		else {
			pad_h_ = conv_param.pad_h();
			pad_w_ = conv_param.pad_w();
		}
		if (!conv_param.has_stride_h()) {
			stride_h_ = stride_w_ = conv_param.stride();
		}
		else {
			stride_h_ = conv_param.stride_h();
			stride_w_ = conv_param.stride_w();
		}

		LayerParameter split_param;
		split_layer.reset(new SplitLayer<Dtype>(split_param));
		split_bottom_vec.clear();
		split_top_vec.clear();
		if (bottom.size() == 2)
			split_bottom_vec.push_back(bottom[1]);
		else
		{
			mask_in_.ReshapeLike(*bottom[0]);
			caffe_set(mask_in_.count(), (Dtype)1.0, mask_in_.mutable_cpu_data());
			split_bottom_vec.push_back(&mask_in_);
		}
		split_top_vec.push_back(&split_top_0);
		split_top_vec.push_back(&split_top_1);
		split_layer->SetUp(split_bottom_vec, split_top_vec);

		LayerParameter eltwise_param;
		eltwise_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
		eltwise_layer.reset(new EltwiseLayer<Dtype>(eltwise_param));
		eltwise_bottom_vec.clear();
		eltwise_top_vec.clear();
		eltwise_bottom_vec.push_back(bottom[0]);
		eltwise_bottom_vec.push_back(split_top_vec[0]);
		eltwise_top_vec.push_back(&eltwise_top);
		eltwise_layer->SetUp(eltwise_bottom_vec, eltwise_top_vec);
	}

	template <typename Dtype>
	void Col2imgMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) 
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
		if (bottom.size() == 2)
		{
			CHECK_EQ(bottom[0]->count(), bottom[1]->count())
				<< "Inputs must have the same dimension.";
		}
		else
		{
			mask_in_.ReshapeLike(*bottom[0]);
			caffe_set(mask_in_.count(), (Dtype)1.0, mask_in_.mutable_cpu_data());
		}
		split_layer->Reshape(split_bottom_vec, split_top_vec);
		eltwise_layer->Reshape(eltwise_bottom_vec, eltwise_top_vec);
		channels_out_ = bottom[0]->channels();
		height_out_ = bottom[0]->height();
		width_out_ = bottom[0]->width();

		channels_ = channels_out_ / kernel_h_ / kernel_w_;
		height_ = stride_h_*(height_out_ - 1) + kernel_h_ - 2 * pad_h_;
		width_ = stride_w_*(width_out_ - 1) + kernel_w_ - 2 * pad_w_;
		top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
		mask_out_.ReshapeLike(*top[0]);
		data_out_.ReshapeLike(*top[0]);
	}

	template <typename Dtype>
	void Col2imgMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		Dtype* mask_out_data = mask_out_.mutable_cpu_data();
		Dtype* data_out_data = data_out_.mutable_cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		split_layer->Forward(split_bottom_vec, split_top_vec);
		eltwise_layer->Forward(eltwise_bottom_vec, eltwise_top_vec);

		const Dtype* eltwise_top_data = eltwise_top_vec[0]->cpu_data();
		const Dtype* mask_in_data = split_top_vec[1]->cpu_data();

		for (int n = 0; n < bottom[0]->num(); ++n) {
			col2im_cpu(eltwise_top_data + eltwise_top_vec[0]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				data_out_data + data_out_.offset(n));
			col2im_cpu(mask_in_data + split_top_vec[1]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				mask_out_data + mask_out_.offset(n));
		}
		caffe_div(top[0]->count(), data_out_data, mask_out_data, top_data);
	}

	template <typename Dtype>
	void Col2imgMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		for (int i = 0; i < split_top_vec.size(); ++i)
			caffe_set(split_top_vec[i]->count(), (Dtype)0, split_top_vec[i]->mutable_cpu_diff());
		for (int i = 0; i < eltwise_top_vec.size(); ++i)
			caffe_set(eltwise_top_vec[i]->count(), (Dtype)0, eltwise_top_vec[i]->mutable_cpu_diff());
		caffe_set(mask_out_.count(), (Dtype)0, mask_out_.mutable_cpu_diff());
		caffe_set(data_out_.count(), (Dtype)0, data_out_.mutable_cpu_diff());

		const Dtype* mask_out_data = mask_out_.cpu_data();
		const Dtype* data_out_data = data_out_.cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();

		Dtype* mask_out_diff = mask_out_.mutable_cpu_diff();
		Dtype* data_out_diff = data_out_.mutable_cpu_diff();
		Dtype* eltwise_top_diff = eltwise_top_vec[0]->mutable_cpu_diff();
		Dtype* mask_in_diff = split_top_vec[1]->mutable_cpu_diff();

		caffe_div(top[0]->count(), top_diff, mask_out_data, data_out_diff);
		caffe_mul(mask_out_.count(), mask_out_data, mask_out_data, mask_out_diff);
		caffe_div(mask_out_.count(), data_out_data, mask_out_diff, mask_out_diff);
		caffe_mul(mask_out_.count(), top_diff, mask_out_diff, mask_out_diff);
		caffe_scal(mask_out_.count(), (Dtype)-1.0, mask_out_diff);

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			im2col_cpu(data_out_diff + data_out_.offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				eltwise_top_diff + eltwise_top_vec[0]->offset(n));
			im2col_cpu(mask_out_diff + mask_out_.offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				mask_in_diff + split_top_vec[1]->offset(n));
		}

		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		if (bottom.size() == 2)
			propagate_down_sub.push_back(propagate_down[1]);
		else
			propagate_down_sub.push_back(propagate_down[0]);
		eltwise_layer->Backward(eltwise_top_vec, propagate_down_sub, eltwise_bottom_vec);
		split_layer->Backward(split_top_vec, propagate_down_sub, split_bottom_vec);
	}

#ifdef CPU_ONLY
	STUB_GPU(Col2imgMaskLayer);
#endif

	INSTANTIATE_CLASS(Col2imgMaskLayer);
	REGISTER_LAYER_CLASS(Col2imgMask);
}