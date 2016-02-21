#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/non_local_2_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void NonLocal2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
		is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
			&& stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
		// Configure output channels and groups.
		channels_ = bottom[0]->channels();

		num_output_ = channels_ * kernel_h_ * kernel_w_;

		num_ = bottom[0]->num();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();

		height_out_ = (height_ + 2 * pad_h_ - kernel_h_)
			/ stride_h_ + 1;
		width_out_ = (width_ + 2 * pad_w_ - kernel_w_)
			/ stride_w_ + 1;

		LayerParameter split_param;
		split_layer_0.reset(new SplitLayer<Dtype>(split_param));
		split_0_top_vec.clear();
		split_0_top_vec.push_back(&split_0_top_0);
		split_0_top_vec.push_back(&split_0_top_1);
		split_layer_0->SetUp(bottom, split_0_top_vec);

		img2col_0_top.Reshape(num_, channels_*kernel_h_*kernel_w_, height_out_, width_out_);
		img2col_1_top.Reshape(num_, channels_*kernel_h_*kernel_w_, height_out_, width_out_);


		euclidean_bottom_0.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_bottom_1.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);

		LayerParameter euclidean_param;
		euclidean_layer.reset(new EuclideanLayer<Dtype>(euclidean_param));
		euclidean_bottom_vec.clear();
		euclidean_top_vec.clear();
		euclidean_bottom_vec.push_back(&euclidean_bottom_0);
		euclidean_bottom_vec.push_back(&euclidean_bottom_1);
		euclidean_top_vec.push_back(&euclidean_top);
		euclidean_layer->SetUp(euclidean_bottom_vec, euclidean_top_vec);

		exp_layer.reset(new ExpLayer<Dtype>(this->layer_param()));
		exp_bottom_vec.clear();
		exp_top_vec.clear();
		exp_bottom_vec.push_back(euclidean_top_vec[0]);
		exp_top_vec.push_back(&exp_top);
		exp_layer->SetUp(exp_bottom_vec, exp_top_vec);

		LayerParameter normalize_param;
		normalize_layer.reset(new NormalizeLayer<Dtype>(normalize_param));
		normalize_bottom_vec.clear();
		normalize_top_vec.clear();
		normalize_bottom.Reshape(num_, kernel_h_*kernel_w_, height_out_, width_out_);
		normalize_bottom_vec.push_back(&normalize_bottom);
		normalize_top_vec.push_back(top[0]);
		normalize_layer->SetUp(normalize_bottom_vec, normalize_top_vec);

	}

	template <typename Dtype>
	void NonLocal2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

		split_layer_0->Reshape(bottom, split_0_top_vec);
		img2col_0_top.Reshape(num_, channels_*kernel_h_*kernel_w_, height_out_, width_out_);
		img2col_1_top.Reshape(num_, channels_*kernel_h_*kernel_w_, height_out_, width_out_);
		euclidean_bottom_0.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_bottom_1.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_layer->Reshape(euclidean_bottom_vec, euclidean_top_vec);
		exp_layer->Reshape(exp_bottom_vec, exp_top_vec);
		normalize_bottom.Reshape(num_, kernel_h_*kernel_w_, height_out_, width_out_);
		normalize_layer->Reshape(normalize_bottom_vec, normalize_top_vec);
	}

	template <typename Dtype>
	void NonLocal2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		split_layer_0->Forward(bottom, split_0_top_vec);

		for (int n = 0; n < num_; ++n)
		{
			im2col_cpu(split_0_top_vec[0]->cpu_data() + split_0_top_vec[0]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				1,1,
				img2col_0_top.mutable_cpu_data() + img2col_0_top.offset(n));

			im2col_center_cpu(split_0_top_vec[1]->cpu_data() + split_0_top_vec[1]->offset(n),
				channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				img2col_1_top.mutable_cpu_data() + img2col_1_top.offset(n));
		}

		euclidean_bottom_vec[0]->ShareData(img2col_0_top);
		euclidean_bottom_vec[1]->ShareData(img2col_1_top);
		euclidean_layer->Forward(euclidean_bottom_vec, euclidean_top_vec);

		caffe_scal(euclidean_top_vec[0]->count(),
			(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_cpu_data());

		exp_layer->Forward(exp_bottom_vec, exp_top_vec);
		normalize_bottom_vec[0]->ShareData(*exp_top_vec[0]);
		normalize_layer->Forward(normalize_bottom_vec, normalize_top_vec);
	}

	template <typename Dtype>
	void NonLocal2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		propagate_down_sub.push_back(propagate_down[0]);
		if (propagate_down[0])
		{
			
			for (int i = 0; i < exp_bottom_vec.size(); i++)
				caffe_set(exp_bottom_vec[i]->count(), (Dtype)0, exp_bottom_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < euclidean_bottom_vec.size(); i++)
				caffe_set(euclidean_bottom_vec[i]->count(), (Dtype)0, euclidean_bottom_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < split_0_top_vec.size(); i++)
				caffe_set(split_0_top_vec[i]->count(), (Dtype)0, split_0_top_vec[i]->mutable_cpu_diff());


			normalize_layer->Backward(normalize_top_vec, propagate_down_sub, normalize_bottom_vec);
			exp_top_vec[0]->ShareDiff(*normalize_bottom_vec[0]);
			exp_layer->Backward(exp_top_vec, propagate_down_sub, exp_bottom_vec);

			caffe_scal(euclidean_top_vec[0]->count(),
				(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_cpu_diff());

			euclidean_layer->Backward(euclidean_top_vec, propagate_down_sub, euclidean_bottom_vec);
			img2col_0_top.ShareDiff(*euclidean_bottom_vec[0]);
			img2col_1_top.ShareDiff(*euclidean_bottom_vec[1]);

			for (int n = 0; n < num_; ++n)
			{
				col2im_center_cpu(img2col_1_top.cpu_diff() + img2col_1_top.offset(n), channels_, height_, width_,
					kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
					split_0_top_vec[1]->mutable_cpu_diff() + split_0_top_vec[1]->offset(n));

				col2im_cpu(img2col_0_top.cpu_diff() + img2col_0_top.offset(n), channels_, height_, width_,
					kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
					1,1,
					split_0_top_vec[0]->mutable_cpu_diff() + split_0_top_vec[0]->offset(n));
			}
			split_layer_0->Backward(split_0_top_vec, propagate_down_sub,bottom);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(NonLocal2Layer);
#endif

	INSTANTIATE_CLASS(NonLocal2Layer);
	REGISTER_LAYER_CLASS(NonLocal2);
}