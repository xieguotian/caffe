#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/non_local_layer.hpp"

namespace caffe{
	template <typename Dtype>
	void NonLocalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

		split_layer_1.reset(new SplitLayer<Dtype>(split_param));
		split_1_bottom_vec.clear();
		split_1_top_vec.clear();
		split_1_bottom_vec.push_back(&img2col_0_top);
		split_1_top_vec.push_back(&split_1_top_0);
		split_1_top_vec.push_back(&split_1_top_1);
		split_layer_1->SetUp(split_1_bottom_vec, split_1_top_vec);

		euclidean_bottom_0.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_bottom_1.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_bottom_0.ShareData(*split_1_top_vec[1]);
		euclidean_bottom_0.ShareDiff(*split_1_top_vec[1]);
		euclidean_bottom_1.ShareData(img2col_1_top);
		euclidean_bottom_1.ShareDiff(img2col_1_top);

		LayerParameter euclidean_param;
		euclidean_layer.reset(new EuclideanLayer<Dtype>(euclidean_param));
		euclidean_bottom_vec.clear();
		euclidean_top_vec.clear();
		euclidean_bottom_vec.push_back(&euclidean_bottom_0);
		euclidean_bottom_vec.push_back(&euclidean_bottom_1);
		euclidean_top_vec.push_back(&euclidean_top);
		euclidean_layer->SetUp(euclidean_bottom_vec, euclidean_top_vec);

		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
		smooth_threshold_layer.reset(new SmoothThresholdLayer<Dtype>(this->layer_param()));
		smooth_bottom_vec.clear();
		smooth_top_vec.clear();
		smooth_bottom_vec.push_back(euclidean_top_vec[0]);
		smooth_top_vec.push_back(&smooth_top);
		smooth_threshold_layer->SetUp(smooth_bottom_vec, smooth_top_vec);
		this->blobs_[0]->ShareData(*smooth_threshold_layer->blobs()[0]);
		this->blobs_[0]->ShareDiff(*smooth_threshold_layer->blobs()[0]);

		split_layer_3.reset(new SplitLayer<Dtype>(split_param));
		split_3_bottom_vec.clear();
		split_3_top_vec.clear();
		split_3_bottom_vec.push_back(smooth_top_vec[0]);
		split_3_top_vec.push_back(&split_3_top_0);
		split_3_top_vec.push_back(&split_3_top_1);
		split_layer_3->SetUp(split_3_bottom_vec, split_3_top_vec);

		LayerParameter normalize_param;
		normalize_layer.reset(new NormalizeLayer<Dtype>(normalize_param));
		normalize_bottom_vec.clear();
		normalize_top_vec.clear();
		normalize_bottom.Reshape(split_3_top_vec[1]->num()*split_3_top_vec[1]->channels(),
			split_3_top_vec[1]->height(), 1, split_3_top_vec[1]->width());
		//normalize_bottom_vec.push_back(split_3_top_vec[1]);
		normalize_bottom_vec.push_back(&normalize_bottom);
		normalize_top_vec.push_back(&normalize_top);
		normalize_layer->SetUp(normalize_bottom_vec, normalize_top_vec);

		split_layer_2.reset(new SplitLayer<Dtype>(split_param));
		split_2_bottom_vec.clear();
		split_2_top_vec.clear();
		split_2_bottom_vec.push_back(&split_2_bottom);
		split_2_bottom_vec[0]->ReshapeLike(*split_1_top_vec[0]);
		split_2_top_vec.push_back(&split_2_top_0);
		split_2_top_vec.push_back(top[0]);
		split_layer_2->SetUp(split_2_bottom_vec, split_2_top_vec);

		LayerParameter eltwise_param;
		eltwise_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
		eltwise_layer.reset(new EltwiseLayer<Dtype>(eltwise_param));
		eltwise_bottom_vec.clear();
		eltwise_top_vec.clear();
		eltwise_bottom_vec.push_back(split_1_top_vec[0]);
		eltwise_bottom_vec.push_back(split_2_top_vec[0]);
		if (top.size() == 3)
		{
			eltwise_top_vec.push_back(top[2]);
			eltwise_layer->SetUp(eltwise_bottom_vec, eltwise_top_vec);
		}
		else
			eltwise_top_vec.push_back(&mask_top);

	}

	template <typename Dtype>
	void NonLocalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
		split_layer_1->Reshape(split_1_bottom_vec, split_1_top_vec);
		euclidean_bottom_0.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_bottom_1.Reshape(num_, channels_, kernel_h_*kernel_w_, height_out_*width_out_);
		euclidean_bottom_0.ShareData(*split_1_top_vec[1]);
		euclidean_bottom_0.ShareDiff(*split_1_top_vec[1]);
		euclidean_bottom_1.ShareData(img2col_1_top);
		euclidean_bottom_1.ShareDiff(img2col_1_top);
		euclidean_layer->Reshape(euclidean_bottom_vec, euclidean_top_vec);
		smooth_threshold_layer->Reshape(smooth_bottom_vec, smooth_top_vec);
		split_layer_3->Reshape(split_3_bottom_vec, split_3_top_vec);
		normalize_bottom.Reshape(split_3_top_vec[1]->num()*split_3_top_vec[1]->channels(),
			split_3_top_vec[1]->height(), 1, split_3_top_vec[1]->width());
		normalize_layer->Reshape(normalize_bottom_vec, normalize_top_vec);
		//top[1]->ReshapeLike(*normalize_top_vec[0]);

		split_2_bottom_vec[0]->ReshapeLike(*split_1_top_vec[0]);
		split_layer_2->Reshape(split_2_bottom_vec, split_2_top_vec);
		top[1]->ReshapeLike(*top[0]);
		//eltwise_bottom_vec[1]->ReshapeLike(*eltwise_bottom_vec[0]);
		if (top.size()==3)
			eltwise_layer->Reshape(eltwise_bottom_vec, eltwise_top_vec);
		//top[0]->ReshapeLike(*euclidean_top_vec[0]);
	}

	template <typename Dtype>
	void NonLocalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

		split_layer_1->Forward(split_1_bottom_vec, split_1_top_vec);
		euclidean_bottom_vec[0]->ShareData(*split_1_top_vec[1]);
		euclidean_layer->Forward(euclidean_bottom_vec, euclidean_top_vec);

		caffe_scal(euclidean_top_vec[0]->count(),
			(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_cpu_data());

		smooth_threshold_layer->Forward(smooth_bottom_vec, smooth_top_vec);
		split_layer_3->Forward(split_3_bottom_vec, split_3_top_vec);
		normalize_bottom_vec[0]->ShareData(*split_3_top_vec[1]);
		normalize_layer->Forward(normalize_bottom_vec, normalize_top_vec);
		//top[1]->ShareData(*normalize_top_vec[0]);
		const Dtype* normalize_top_data = normalize_top_vec[0]->cpu_data();
		const Dtype* split_3_top_data_1 = normalize_top_vec[0]->cpu_data();
		Dtype* top_1_data = top[1]->mutable_cpu_data();
		const int norm_offset = normalize_top_vec[0]->offset(1);

		for (int n = 0; n < normalize_top_vec[0]->num(); ++n)
		{
			for (int ch = 0; ch < channels_; ++ch)
			{
				caffe_copy(norm_offset, split_3_top_data_1, top_1_data);
				top_1_data += norm_offset;
			}
			split_3_top_data_1 += norm_offset;
		}

		//int tmp_offset = smooth_top_vec[0]->count() / smooth_top_vec[0]->num();
		const int tmp_offset = split_3_top_vec[0]->offset(1);
		
		Dtype* split_2_bottom_data = split_2_bottom_vec[0]->mutable_cpu_data();
		//const Dtype* smooth_top_data = smooth_top_vec[0]->cpu_data();
		const Dtype* split_3_top_data = split_3_top_vec[0]->cpu_data();
		for (int n = 0; n < split_2_bottom_vec[0]->num(); ++n)
		{
			for (int ch = 0; ch < channels_; ++ch)
			{
				//caffe_copy(tmp_offset, smooth_top_data, split_2_bottom_data);
				caffe_copy(tmp_offset, split_3_top_data, split_2_bottom_data);
				split_2_bottom_data += tmp_offset;
			}
			//smooth_top_data += smooth_top_vec[0]->offset(1);
			split_3_top_data += tmp_offset;
		}

		split_layer_2->Forward(split_2_bottom_vec, split_2_top_vec);
		if (top.size() == 3)
			eltwise_layer->Forward(eltwise_bottom_vec, eltwise_top_vec);
	
	}

	template <typename Dtype>
	void NonLocalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		propagate_down_sub.push_back(propagate_down[0]);
		if (propagate_down[0])
		{
			for (int i = 0; i < eltwise_bottom_vec.size(); i++)
				caffe_set(eltwise_bottom_vec[i]->count(), (Dtype)0, eltwise_bottom_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < smooth_bottom_vec.size(); i++)
				caffe_set(smooth_bottom_vec[i]->count(), (Dtype)0, smooth_bottom_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < euclidean_bottom_vec.size(); i++)
				caffe_set(euclidean_bottom_vec[i]->count(), (Dtype)0, euclidean_bottom_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < split_1_bottom_vec.size(); i++)
				caffe_set(split_1_bottom_vec[i]->count(), (Dtype)0, split_1_bottom_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < smooth_top_vec.size(); i++)
				caffe_set(smooth_top_vec[i]->count(), (Dtype)0, smooth_top_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < split_0_top_vec.size(); i++)
				caffe_set(split_0_top_vec[i]->count(), (Dtype)0, split_0_top_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < split_3_top_vec.size(); i++)
				caffe_set(split_3_top_vec[i]->count(), (Dtype)0, split_3_top_vec[i]->mutable_cpu_diff());
			for (int i = 0; i < normalize_top_vec.size(); i++)
				caffe_set(normalize_top_vec[i]->count(), (Dtype)0, normalize_top_vec[i]->mutable_cpu_diff());

			if (top.size() == 3)
				eltwise_layer->Backward(eltwise_top_vec, propagate_down_sub, eltwise_bottom_vec);

			split_layer_2->Backward(split_2_top_vec, propagate_down_sub, split_2_bottom_vec);
			//int tmp_offset = smooth_top_vec[0]->offset(1);
			const int tmp_offset = split_3_top_vec[0]->offset(1);
			//const Dtype* eltwise_bottom_1_diff = eltwise_bottom_vec[1]->cpu_diff();
			const Dtype* split_2_bottom_diff = split_2_bottom_vec[0]->cpu_diff();
			//Dtype* smooth_top_diff = smooth_top_vec[0]->mutable_cpu_diff();
			Dtype* split_3_top_diff = split_3_top_vec[0]->mutable_cpu_diff();
			for (int n = 0; n < split_2_bottom_vec[0]->num(); ++n)
			{
				for (int ch = 0; ch < channels_; ++ch)
				{
					//caffe_add(tmp_offset, smooth_top_diff, split_2_bottom_diff, smooth_top_diff);
					caffe_add(tmp_offset, split_3_top_diff, split_2_bottom_diff, split_3_top_diff);
					split_2_bottom_diff += tmp_offset;
				}
				//smooth_top_diff += tmp_offset;
				split_3_top_diff += tmp_offset;
			}

			const int norm_offset = normalize_top_vec[0]->offset(1);
			Dtype* normalize_diff = normalize_top_vec[0]->mutable_cpu_diff();
			const Dtype* top_1_diff = top[1]->cpu_diff();
			for (int n = 0; n < normalize_top_vec[0]->num(); ++n)
			{
				for (int ch = 0; ch < channels_; ++ch)
				{
					caffe_add(tmp_offset, normalize_diff, top_1_diff, normalize_diff);
					top_1_diff += norm_offset;
				}
				normalize_diff += norm_offset;
			}
			//nomralize_top_vec[0]->ShareDiff(*top[1]);
			normalize_layer->Backward(normalize_top_vec, propagate_down_sub, normalize_bottom_vec);
			split_3_top_vec[1]->ShareDiff(*normalize_bottom_vec[0]);
			split_layer_3->Backward(split_3_top_vec, propagate_down_sub, split_3_bottom_vec);
			smooth_threshold_layer->Backward(smooth_top_vec, propagate_down_sub, smooth_bottom_vec);

			caffe_scal(euclidean_top_vec[0]->count(),
				(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_cpu_diff());

			euclidean_layer->Backward(euclidean_top_vec, propagate_down_sub, euclidean_bottom_vec);
			split_1_top_vec[1]->ShareDiff(*euclidean_bottom_vec[0]);
			split_layer_1->Backward(split_1_top_vec, propagate_down_sub, split_1_bottom_vec);

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
	STUB_GPU(NonLocalLayer);
#endif

	INSTANTIATE_CLASS(NonLocalLayer);
	REGISTER_LAYER_CLASS(NonLocal);
}