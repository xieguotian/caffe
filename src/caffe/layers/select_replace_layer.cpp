#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/select_replace_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void SelectReplaceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
		// Configure the kernel size, padding, stride, and inputs.
		ConvolutionParameter conv_param = this->layer_param_.convolution_param();
		/*CHECK(!conv_param.has_kernel_size() !=
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
		}*/
		//kernel
		if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
			CHECK_EQ(0, conv_param.kernel_size_size())
				<< "Either kernel_size or kernel_h/w should be specified; not both.";
			kernel_h_ = conv_param.kernel_h();
			kernel_w_ = conv_param.kernel_w();
		}
		else {
			const int num_kernel_dims = conv_param.kernel_size_size();
			CHECK(num_kernel_dims == 1)
				<< "kernel_size must be specified once, or once per spatial dimension "
				<< "(kernel_size specified " << num_kernel_dims << " times; ";

			kernel_h_ = kernel_w_ = conv_param.kernel_size(0);
		}
		CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
		CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
		//stride
		if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
			CHECK_EQ(0, conv_param.stride_size())
				<< "Either stride or stride_h/w should be specified; not both.";
			stride_h_ = conv_param.stride_h();
			stride_w_ = conv_param.stride_w();
		}
		else {
			const int num_stride_dims = conv_param.stride_size();
			CHECK(num_stride_dims == 0 || num_stride_dims == 1)
				<< "stride must be specified once, or once per spatial dimension "
				<< "(stride specified " << num_stride_dims << " times; ";
			const int kDefaultStride = 1;
			stride_h_ = stride_w_ = (num_stride_dims == 0) ? kDefaultStride : conv_param.stride(0);
		}
		//pad
		if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
			CHECK_EQ(0, conv_param.pad_size())
				<< "Either pad or pad_h/w should be specified; not both.";
			pad_h_ = conv_param.pad_h();
			pad_w_ = conv_param.pad_w();
		}
		else {
			const int num_pad_dims = conv_param.pad_size();
			CHECK(num_pad_dims == 0 || num_pad_dims == 1)
				<< "pad must be specified once, or once per spatial dimension "
				<< "(pad specified " << num_pad_dims << " times; ";
			const int kDefaultPad = 0;
			pad_h_ = pad_w_ = (num_pad_dims == 0) ? kDefaultPad : conv_param.pad(0);
		}

		// Special case: im2col is the identity for 1x1 convolution with stride 1
		// and no padding, so flag for skipping the buffer and transformation.
		//is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
		//	&& stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
		// Configure output channels and groups.
		//channels_ = bottom[0]->channels();

		//num_output_ = channels_ * kernel_h_ * kernel_w_;
		//num_output_ = kernel_h_ * kernel_w_;

		//num_ = bottom[0]->num();
		//height_ = bottom[0]->height();
		//width_ = bottom[0]->width();

		//height_out_ = (height_ + 2 * pad_h_ - kernel_h_)
		//	/ stride_h_ + 1;
		//width_out_ = (width_ + 2 * pad_w_ - kernel_w_)
		//	/ stride_w_ + 1;
		num_ = bottom[0]->num();
		height_out_ = bottom[0]->height();
		width_out_ = bottom[0]->width();
		channels_ = bottom[0]->channels() / bottom[1]->channels();
		height_ = stride_h_*(height_out_ - 1) + kernel_h_ - 2 * pad_h_;
		width_ = stride_w_*(width_out_ - 1) + kernel_w_ - 2 * pad_w_;
		top_N_ = bottom[1]->channels();
	}

	template <typename Dtype>
	void SelectReplaceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		num_ = bottom[0]->num();
		height_out_ = bottom[0]->height();
		width_out_ = bottom[0]->width();
		channels_ = bottom[0]->channels() / bottom[1]->channels();
		height_ = stride_h_*(height_out_ - 1) + kernel_h_ - 2 * pad_h_;
		width_ = stride_w_*(width_out_ - 1) + kernel_w_ - 2 * pad_w_;
		top_N_ = bottom[1]->channels();
		top[0]->Reshape(num_, channels_, height_, width_);
		//count_coef_.ReshapeLike(*bottom[1]);
		count_coef_.Reshape(num_, 1, height_, width_);
		if (stride_h_ != 1 || stride_w_ != 1)
		{
			CHECK_EQ(2, top.size())
				<< "stride is not 1, top size must be 2.";

			top[1]->Reshape(num_, channels_, height_, width_);
		}

		idx_trans_cache_.ReshapeLike(*bottom[1]);
	}


	template <typename Dtype>
	void select_replace(const Dtype* in_data, const Dtype* index_data, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int top_N, 
		Dtype* out_data, Dtype* count_coef)
	{
		int channels_col = channels*top_N;
		
		for (int c_col = 0; c_col < channels_col; ++c_col) 
		{
			int top_N_index = c_col % top_N;
			int c_im = c_col / top_N;
			for (int h_col = 0; h_col < height_col; ++h_col) 
			{
				for (int w_col = 0; w_col < width_col; ++w_col) 
				{
					int idx = (top_N_index*height_col + h_col)*width_col + w_col;
					int d_idx = index_data[idx];
					//===================
					int h_in = h_col * stride_h - pad_h + (int)(d_idx / kernel_w%kernel_h);
					int w_in = w_col* stride_w - pad_w + (int)(d_idx%kernel_w);
					//===================
					//if ( d_idx >= 0)
					if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
					{
						//int w_in = d_idx % width;
						//int h_in = d_idx / width % height;
						int out_index = (c_im*height + h_in)*width + w_in;
						out_data[out_index] +=
							in_data[(c_col*height_col + h_col)*width_col + w_col];

						if (c_im == 0)
						{
							count_coef[h_in*width + w_in] += 1;
						}
					}
				}
			}
		}


		for (int ch = 0; ch < channels; ++ch)
		{
			for (int h = 0; h < height; ++h)
			{
				for (int w = 0; w < width; ++w)
				{
					int count = count_coef[h*width + w];
					if (count != 0)
						out_data[(ch*height + h)*width + w] /= count;
				}
			}
		}
	}

	template <typename Dtype>
	void SelectReplaceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		caffe_set(top[0]->count(), (Dtype)0, top[0]->mutable_cpu_data());
		caffe_set(count_coef_.count(), (Dtype)0, count_coef_.mutable_cpu_data());
		for (int n = 0; n < num_; ++n)
		{
			select_replace<Dtype>(bottom[0]->cpu_data() + bottom[0]->offset(n),
				bottom[1]->cpu_data() + bottom[1]->offset(n),
				channels_, height_, width_, kernel_h_, kernel_w_,
				pad_h_, pad_w_, stride_h_, stride_w_,
				height_out_, width_out_, top_N_,
				top[0]->mutable_cpu_data()+top[0]->offset(n),
				count_coef_.mutable_cpu_data() + count_coef_.offset(n));

			if (top.size() == 2)
			{
				Dtype* top_1_data = top[1]->mutable_cpu_data() + top[1]->offset(n);
				const Dtype* count_coef_data = count_coef_.cpu_data() + count_coef_.offset(n);
				int tmp_offset = count_coef_.offset(1);
				for (int i = 0; i < tmp_offset; ++i)
					top_1_data[i] = (count_coef_data[i]>0) ? 1 : 0;

				for (int ch = 1; ch < channels_; ++ch)
					caffe_copy(tmp_offset, top_1_data, top_1_data + ch*tmp_offset);
			}
		}
	}

	template <typename Dtype>
	void SelectReplaceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SelectReplaceLayer);
#endif

	INSTANTIATE_CLASS(SelectReplaceLayer);
	REGISTER_LAYER_CLASS(SelectReplace);
}