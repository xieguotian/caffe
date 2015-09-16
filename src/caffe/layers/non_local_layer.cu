#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe 
{

	template <typename Dtype>
	__global__ void im2col_center_gpu_kernel(const int n, const Dtype* data_im,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int height_col, const int width_col,
		Dtype* data_col) {
		CUDA_KERNEL_LOOP(index, n) {
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;
			int channel_in = h_index / height_col;
			int channel_out = channel_in * kernel_h * kernel_w;
			int h_in = h_out * stride_h - pad_h;
			int w_in = w_out * stride_w - pad_w;
			Dtype* data_col_ptr = data_col;
			data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
			const Dtype* data_im_ptr = data_im;
			data_im_ptr += (channel_in * height + h_in) * width + w_in;
			int h_offset = kernel_h / 2;
			int w_offset = kernel_w / 2;
			for (int i = 0; i < kernel_h; ++i) {
				for (int j = 0; j < kernel_w; ++j) {
					int h = h_in + h_offset;
					int w = w_in + w_offset;

					*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
						data_im_ptr[h_offset * width + w_offset] : 0;
					data_col_ptr += height_col * width_col;
				}
			}
		}
	}

	template <typename Dtype>
	void im2col_center_gpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		Dtype* data_col) {
		// We are going to launch channels * height_col * width_col kernels, each
		// kernel responsible for copying a single-channel grid.
		int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
		int num_kernels = channels * height_col * width_col;
		// NOLINT_NEXT_LINE(whitespace/operators)
		im2col_center_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels),
			CAFFE_CUDA_NUM_THREADS >> >(
			num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
			pad_w, stride_h, stride_w, height_col,
			width_col, data_col);
		CUDA_POST_KERNEL_CHECK;
	}


	// Explicit instantiation
	template void im2col_center_gpu<float>(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		float* data_col);
	template void im2col_center_gpu<double>(const double* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		double* data_col);

	template <typename Dtype>
	__global__ void col2im_center_gpu_kernel(const int n, const Dtype* data_col,
		const int height, const int width, const int channels,
		const int patch_h, const int patch_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int height_col, const int width_col,
		Dtype* data_im) {
		CUDA_KERNEL_LOOP(index, n) {
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;
			int channel_in = h_index / height_col;
			int channel_out = channel_in*patch_h*patch_w;
			int h_in = h_out * stride_h - pad_h;
			int w_in = w_out * stride_w - pad_w;
			const Dtype* data_col_ptr = data_col;
			data_col_ptr += (channel_out*height_col + h_out)*width_col + w_out;

			int mid_h = patch_h / 2;
			int mid_w = patch_w / 2;
			h_in += mid_h;
			w_in += mid_w;
			Dtype* data_im_ptr = data_im;
			data_im_ptr += (channel_in*height + h_in)*width + w_in;
			for (int i = 0; i < patch_h; ++i)
			{
				for (int j = 0; j < patch_w; ++j)
				{
					*data_im_ptr += *data_col_ptr;
					data_col_ptr += height_col*width_col;
				}
			}
		}
	}

	template <typename Dtype>
	void col2im_center_gpu(const Dtype* data_col, const int channels,
		const int height, const int width, const int patch_h, const int patch_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, Dtype* data_im) {
		int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
		int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
		//int num_kernels = channels * height * width;
		int num_kernels = channels * height_col * width_col;
		// To avoid involving atomic operations, we will launch one kernel per
		// bottom dimension, and then in the kernel add up the top dimensions.
		// NOLINT_NEXT_LINE(whitespace/operators)
		col2im_center_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels),
			CAFFE_CUDA_NUM_THREADS >> >(
			num_kernels, data_col, height, width, channels, patch_h, patch_w,
			pad_h, pad_w, stride_h, stride_w,
			height_col, width_col, data_im);
		CUDA_POST_KERNEL_CHECK;
	}

	// Explicit instantiation
	template void col2im_center_gpu<float>(const float* data_col, const int channels,
		const int height, const int width, const int patch_h, const int patch_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, float* data_im);
	template void col2im_center_gpu<double>(const double* data_col, const int channels,
		const int height, const int width, const int patch_h, const int patch_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, double* data_im);

	template <typename Dtype>
	void NonLocalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		split_layer_0->Forward(bottom, split_0_top_vec);

		for (int n = 0; n < num_; ++n)
		{
			im2col_gpu(split_0_top_vec[0]->gpu_data() + split_0_top_vec[0]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				img2col_0_top.mutable_gpu_data() + img2col_0_top.offset(n));

			im2col_center_gpu(split_0_top_vec[1]->gpu_data() + split_0_top_vec[1]->offset(n),
				channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				img2col_1_top.mutable_gpu_data() + img2col_1_top.offset(n));
		}

		split_layer_1->Forward(split_1_bottom_vec, split_1_top_vec);
		euclidean_bottom_0.ShareData(*split_1_top_vec[1]);
		euclidean_layer->Forward(euclidean_bottom_vec, euclidean_top_vec);

		caffe_gpu_scal(euclidean_top_vec[0]->count(),
			(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_gpu_data());

		smooth_threshold_layer->Forward(smooth_bottom_vec, smooth_top_vec);

		int tmp_offset = smooth_top_vec[0]->count() / smooth_top_vec[0]->num();
		//Dtype* eltwise_bottom_1_data = eltwise_bottom_vec[1]->mutable_gpu_data();
		Dtype* split_2_bottom_data = split_2_bottom_vec[0]->mutable_gpu_data();
		const Dtype* smooth_top_data = smooth_top_vec[0]->gpu_data();
		for (int n = 0; n < split_2_bottom_vec[0]->num(); ++n)
		{
			for (int ch = 0; ch < channels_; ++ch)
			{
				caffe_copy(tmp_offset, smooth_top_data, split_2_bottom_data);
				split_2_bottom_data += tmp_offset;
			}
			smooth_top_data += smooth_top_vec[0]->offset(1);
		}

		split_layer_2->Forward(split_2_bottom_vec, split_2_top_vec);
		if (top.size() == 2)
			eltwise_layer->Forward(eltwise_bottom_vec, eltwise_top_vec);

	}

	template <typename Dtype>
	void NonLocalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		propagate_down_sub.push_back(propagate_down[0]);
		if (propagate_down[0])
		{
			for (int i = 0; i < eltwise_bottom_vec.size(); i++)
				caffe_gpu_set(eltwise_bottom_vec[i]->count(), (Dtype)0, eltwise_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < smooth_bottom_vec.size(); i++)
				caffe_gpu_set(smooth_bottom_vec[i]->count(), (Dtype)0, smooth_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < eltwise_bottom_vec.size(); i++)
				caffe_gpu_set(eltwise_bottom_vec[i]->count(), (Dtype)0, eltwise_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < split_1_bottom_vec.size(); i++)
				caffe_gpu_set(split_1_bottom_vec[i]->count(), (Dtype)0, split_1_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < smooth_top_vec.size(); i++)
				caffe_gpu_set(smooth_top_vec[i]->count(), (Dtype)0, smooth_top_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < split_0_top_vec.size(); i++)
				caffe_gpu_set(split_0_top_vec[i]->count(), (Dtype)0, split_0_top_vec[i]->mutable_gpu_diff());
			if (top.size() == 2)
				eltwise_layer->Backward(eltwise_top_vec, propagate_down_sub, eltwise_bottom_vec);

			split_layer_2->Backward(split_2_top_vec, propagate_down_sub, split_2_bottom_vec);
			int tmp_offset = smooth_top_vec[0]->offset(1);
			//const Dtype* eltwise_bottom_1_diff = eltwise_bottom_vec[1]->gpu_diff();
			const Dtype* split_2_bottom_diff = split_2_bottom_vec[0]->gpu_diff();
			Dtype* smooth_top_diff = smooth_top_vec[0]->mutable_gpu_diff();
			for (int n = 0; n < split_2_bottom_vec[0]->num(); ++n)
			{
				for (int ch = 0; ch < channels_; ++ch)
				{
					caffe_gpu_add(tmp_offset, smooth_top_diff, split_2_bottom_diff, smooth_top_diff);
					split_2_bottom_diff += tmp_offset;
				}
				smooth_top_diff += tmp_offset;
			}
			smooth_threshold_layer->Backward(smooth_top_vec, propagate_down_sub, smooth_bottom_vec);

			caffe_gpu_scal(euclidean_top_vec[0]->count(),
				(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_gpu_diff());

			euclidean_layer->Backward(euclidean_top_vec, propagate_down_sub, euclidean_bottom_vec);
			split_layer_1->Backward(split_1_top_vec, propagate_down_sub, split_1_bottom_vec);

			for (int n = 0; n < num_; ++n)
			{
				col2im_center_gpu(img2col_1_top.gpu_diff() + img2col_1_top.offset(n), channels_, height_, width_,
					kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
					split_0_top_vec[1]->mutable_gpu_diff() + split_0_top_vec[1]->offset(n));

				col2im_gpu(img2col_0_top.gpu_diff() + img2col_0_top.offset(n), channels_, height_, width_,
					kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
					split_0_top_vec[0]->mutable_gpu_diff() + split_0_top_vec[0]->offset(n));
			}
			split_layer_0->Backward(split_0_top_vec, propagate_down_sub, bottom);

			CUDA_POST_KERNEL_CHECK;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NonLocalLayer);
}