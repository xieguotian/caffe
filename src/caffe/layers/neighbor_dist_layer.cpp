#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void NeighborDistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
		channels_ = bottom[0]->channels();

		//num_output_ = channels_ * kernel_h_ * kernel_w_;
		num_output_ = kernel_h_ * kernel_w_;

		num_ = bottom[0]->num();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();

		height_out_ = (height_ + 2 * pad_h_ - kernel_h_)
			/ stride_h_ + 1;
		width_out_ = (width_ + 2 * pad_w_ - kernel_w_)
			/ stride_w_ + 1;

	}

	template <typename Dtype>
	void NeighborDistLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
		num_ = bottom[0]->num();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
			" convolution kernel.";

		height_out_ = (height_ + 2 * pad_h_ - kernel_h_)
			/ stride_h_ + 1;
		width_out_ = (width_ + 2 * pad_w_ - kernel_w_)
			/ stride_w_ + 1;

		top[0]->Reshape(num_, num_output_, height_out_, width_out_);
	}

	template <typename Dtype>
	void neighbor_dist_cpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,const int stride_h, 
		const int stride_w, const int height_col, const int width_col, const Dtype fill_data,
		Dtype* data_col) {

		int channels_col = kernel_h * kernel_w;
		int tmp_offset = height*width;

		for (int h_out = 0; h_out < height_col; ++h_out)
		{
			for (int w_out = 0; w_out < width_col; ++w_out)
			{
				//int h_in_center = h_out*stride_h - pad_h + (int)(height / 2.0);
				//int w_in_center = w_out*stride_w - pad_w + (int)(width / 2.0);
				int h_in_center = h_out*stride_h - pad_h + (int)(kernel_h / 2.0);
				int w_in_center = w_out*stride_w - pad_w + (int)(kernel_w / 2.0);

				for (int ch_out = 0; ch_out < channels_col; ++ch_out)
				{
					int w_in = ch_out % kernel_w;
					int h_in = (ch_out / kernel_w) % kernel_h;
					h_in += h_out*stride_h - pad_h;
					w_in += w_out*stride_w - pad_w;

					Dtype* data_col_ptr = data_col +
						(ch_out*height_col + h_out)*width_col + w_out;
					if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width)
					{
						const Dtype* data_im_ptr = data_im + h_in*width + w_in;
						const Dtype* data_im_center_ptr = data_im +
							h_in_center*width + w_in_center;

						for (int ch = 0; ch < channels; ++ch)
						{
							data_col_ptr[0] += (data_im_ptr[0] - data_im_center_ptr[0])*
								(data_im_ptr[0] - data_im_center_ptr[0]);
							data_im_ptr += tmp_offset;
							data_im_center_ptr += tmp_offset;
						}
						data_col_ptr[0] = std::sqrt(data_col_ptr[0]);
					}
					else
					{
						data_col_ptr[0] = fill_data;
					}
				}
			}
		}


	}

	template <typename Dtype>
	void NeighborDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Dtype max_num = -1;//std::numeric_limits<Dtype>::max() - 1;
		caffe_set(top[0]->count(), (Dtype)0, top[0]->mutable_cpu_data());
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			neighbor_dist_cpu(bottom[0]->cpu_data() + bottom[0]->offset(n),
				channels_, height_, width_, kernel_h_, kernel_w_,
				pad_h_, pad_w_, stride_h_, stride_w_, height_out_, width_out_,max_num,
				top[0]->mutable_cpu_data() + top[0]->offset(n));
		}
	}

	template <typename Dtype>
	void NeighborDistLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(NeighborDistLayer);
#endif

	INSTANTIATE_CLASS(NeighborDistLayer);
	REGISTER_LAYER_CLASS(NeighborDist);
}