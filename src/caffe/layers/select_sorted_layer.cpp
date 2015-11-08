#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

#include "boost\iterator\zip_iterator.hpp"
#include "boost\tuple\tuple.hpp"
#include "boost\tuple\tuple_comparison.hpp"

namespace caffe{
	template <typename Dtype>
	void SelectSortedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

		top_N_ = this->layer_param_.k_sparse_param().sparse_k();
		CHECK(top_N_ >= 1) << "top_N_ must be greater than 1";

		nei_dist_layer.reset(new NeighborDistLayer<Dtype>(this->layer_param()));
		nei_dist_bottom_vec.clear();
		nei_dist_top_vec.clear();
		nei_dist_bottom_vec.push_back(bottom[0]);
		nei_dist_top_vec.push_back(&dist);
		nei_dist_layer->SetUp(nei_dist_bottom_vec, nei_dist_top_vec);
	}

	template <typename Dtype>
	void SelectSortedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

		nei_dist_layer->Reshape(nei_dist_bottom_vec, nei_dist_top_vec);
		key.ReshapeLike(dist);
		index.ReshapeLike(dist);

		top[0]->Reshape(num_, top_N_, height_out_, width_out_);
		if (top.size() == 2)
		{
			top[1]->Reshape(num_, channels_*top_N_, height_out_, width_out_);
		}
	}


	template <typename Dtype>
	void select_top_N(const Dtype* in_data, const Dtype* index_data,const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,const int stride_h, const int stride_w, 
		const int height_col, const int width_col, const int top_N, Dtype* out_data)
	{
		for (int c_col = 0; c_col < channels*top_N; ++c_col)
		{
			int top_N_idx = c_col % top_N;
			int c_im = c_col / top_N;
			for (int h_col = 0; h_col < height_col; ++h_col)
			{
				for (int w_col = 0; w_col < width_col; ++w_col)
				{

					int idx = (top_N_idx*height_col + h_col)*width_col + w_col;
					int d_idx = index_data[idx];
					//=============
					int h_in = h_col * stride_h - pad_h + (int)(d_idx / kernel_w%kernel_h);
					int w_in = w_col* stride_w - pad_w + (int)(d_idx%kernel_w);
					//=============
					//if (d_idx >= 0)
					if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
					{
						//int w_in = d_idx % width;
						//int h_in = d_idx / width % height;

						out_data[(c_col*height_col + h_col)*width_col + w_col] =
							in_data[(c_im*height + h_in)*width + w_in];
					}
					else
					{
						out_data[(c_col*height_col + h_col)*width_col + w_col] = 0;
					}

				}
			}
		}
	}

	template<typename Dtype>
	void sort_top_N_dist(const Dtype* in_dist, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int top_N,
		Dtype* out_index)
	{
		int channels_col = kernel_h*kernel_w;
		vector<pair<Dtype, int>> dist_list(channels_col);
		
		for (int h_col = 0; h_col < height_col; ++h_col)
		{
			for (int w_col = 0; w_col < width_col; ++w_col)
			{
				const Dtype* dist_ptr = in_dist + h_col*width_col + w_col;

				for (int ch_col = 0; ch_col < channels_col; ++ch_col)
				{
					dist_list[ch_col].first = dist_ptr[0];
					dist_list[ch_col].second = ch_col;
					dist_ptr += height_col*width_col;
				}
				std::stable_sort(dist_list.begin(), dist_list.end());

				Dtype* out_index_ptr = out_index + h_col*width_col + w_col;

				//int w_in_base = w_col*stride_w - pad_w;
				//int h_in_base = h_col*stride_h - pad_h;
				for (int ch_col = 0; ch_col < top_N; ++ch_col)
				{
					out_index_ptr[0] = dist_list[ch_col].second;
					/*int w_in = w_in_base + dist_list[ch_col].second % kernel_w;
					int h_in = h_in_base + (int)(dist_list[ch_col].second / kernel_w);
					if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
					{
						out_index_ptr[0] = h_in*width + w_in;
					}
					else
					{
						out_index_ptr[0] = -1;
					}*/
					out_index_ptr += height_col*width_col;
				}
			}
		}
	}

	template <typename Dtype>
	void SelectSortedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		nei_dist_layer->Forward(nei_dist_bottom_vec, nei_dist_top_vec);
		Dtype* dist_data = dist.mutable_cpu_data();
		
		for (int i = 0; i < dist.count(); ++i)
		{
			if (dist_data[i] < 0)
				dist_data[i] = std::numeric_limits<Dtype>::max();
		}

		for (int n = 0; n < num_; ++n)
		{
			sort_top_N_dist<Dtype>(dist.cpu_data() + dist.offset(n),
				channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
				stride_h_, stride_w_, height_out_, width_out_, top_N_,
				top[0]->mutable_cpu_data() + top[0]->offset(n));

			if (top.size() == 2)
			{
				select_top_N<Dtype>(bottom[1]->cpu_data() + bottom[1]->offset(n),
					top[0]->cpu_data() + top[0]->offset(n),
					channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
					stride_h_, stride_w_, height_out_, width_out_, top_N_,
					top[1]->mutable_cpu_data() + top[1]->offset(n));
			}
		}


		
	}

	template <typename Dtype>
	void SelectSortedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			
		}

		if (top.size() == 2 && propagate_down[1])
		{

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SelectSortedLayer);
#endif

	INSTANTIATE_CLASS(SelectSortedLayer);
	REGISTER_LAYER_CLASS(SelectSorted);
}