#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe 
{
	template <typename Dtype>
	__global__ void neighbor_dist_gpu_kernel(const int n, const Dtype* data_im,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels,const Dtype fill_data,
		Dtype* data_col) {
		CUDA_KERNEL_LOOP(index, n) {
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;

			int w_in_index = h_index / height_col;
			int w_in = w_in_index % kernel_w;
			int h_in_index = w_in_index / kernel_w;
			int h_in = h_in_index % kernel_h;
			int ch_out_idx = w_in_index;

			h_in += h_out * stride_h - pad_h;
			w_in += w_out* stride_w - pad_w;

			//int h_in_center = h_out * stride_h - pad_h + (int)(height / 2.0);
			//int w_in_center = w_out* stride_w - pad_w + (int)(width / 2.0);
			int h_in_center = h_out * stride_h - pad_h + (int)(kernel_h / 2.0);
			int w_in_center = w_out* stride_w - pad_w + (int)(kernel_w / 2.0);

			if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width) 
			{

				const Dtype* data_im_ptr = data_im + 
					h_in*width + w_in;
				const Dtype* data_im_center_ptr = data_im +
					h_in_center*width + w_in_center;

				int tmp_offset = height*width;
				for (int ch = 0; ch < channels; ++ch)
				{
					data_col[index] += (data_im_ptr[0] - data_im_center_ptr[0])*(data_im_ptr[0] - data_im_center_ptr[0]);
					data_im_ptr += tmp_offset;
					data_im_center_ptr += tmp_offset;
				}
				data_col[index] = std::sqrt(data_col[index]);
			}
			else
			{
				data_col[index] = fill_data;
			}
		}
	}

	template <typename Dtype>
	void NeighborDistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int num_kernels = kernel_h_*kernel_w_*height_out_ * width_out_;

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		Dtype max_num = -1;//std::numeric_limits<Dtype>::max() - 1;

		caffe_gpu_set(top[0]->count(), (Dtype)0, top[0]->mutable_gpu_data());
		for (int n = 0; n < num_; ++n)
		{

			neighbor_dist_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
				num_kernels, bottom_data + bottom[0]->offset(n), height_, width_,
				kernel_h_, kernel_w_, pad_h_,
				pad_w_, stride_h_, stride_w_, height_out_,
				width_out_, channels_, max_num,
				top_data + top[0]->offset(n));
			CUDA_POST_KERNEL_CHECK;
		}
	}

	template <typename Dtype>
	__global__ void neighbor_dist_backward_gpu_kernel(const int n, const Dtype* in_diff, 
		const Dtype* in_data,const Dtype* out_data,
		const int height, const int width, const int channels,
		const int patch_h, const int patch_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int height_col, const int width_col,
		Dtype* out_diff) {
		CUDA_KERNEL_LOOP(index, n) {
			Dtype val = 0;
			int w = index % width + pad_w;
			int h = (index / width) % height + pad_h;
			int c = index / (width * height);
			// compute the start and end of the output
			int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
			int w_col_end = min(w / stride_w + 1, width_col);
			int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
			int h_col_end = min(h / stride_h + 1, height_col);


			int offset = (h * patch_w + w) * height_col * width_col;
			int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
			int coeff_w_col = (1 - stride_w * height_col * width_col);

			for (int h_col = h_col_start; h_col < h_col_end; ++h_col) 
			{
				for (int w_col = w_col_start; w_col < w_col_end; ++w_col) 
				{
					int w_center = w_col*stride_w - pad_w + (int)(patch_w / 2.0);
					int h_center = h_col*stride_h - pad_h + (int)(patch_h / 2.0);
					int idx_center = (c*height + h_center)*width + w_center;

					int out_idx = offset + h_col * coeff_h_col + w_col * coeff_w_col;
					if (out_data[out_idx] != 0)
					{
						val += in_diff[out_idx] * (in_data[index] - in_data[idx_center]) / out_data[out_idx];
					}
				}
			}

			int w_col = (w - (int)(patch_w / 2.0)) / stride_w;
			int h_col = (h - (int)(patch_h / 2.0)) / stride_h;

			for (int k_h = 0; k_h < patch_h; k_h++)
			{
				int h_in = k_h + h_col*stride_h - pad_h;
				for (int k_w = 0; k_w < patch_w; k_w++)
				{
					int w_in = k_w + w_col*stride_w - pad_w;
					int out_idx = ((k_h*patch_w + k_w)*height_col + h_col)*width_col + w_col;

					if (h_in >= 0 && w_in >= 0 && h_in < height && w_in < width
						&& out_data[out_idx] != 0 /*&& k_h != (int)(patch_h / 2.0) && k_w != (int)(patch_w / 2.0)*/)  // why
					{
						int d_idx = (c*height + h_in)*width + w_in;
						val -= in_diff[out_idx] / out_data[out_idx] * (in_data[d_idx] - in_data[index]);
					}
				}
			}

			out_diff[index] = val;
		}
	}

	template <typename Dtype>
	void NeighborDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			int num_kernels = channels_*height_*width_;
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				neighbor_dist_backward_gpu_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
					num_kernels, top[0]->gpu_diff() + top[0]->offset(n),
					bottom[0]->gpu_data() + bottom[0]->offset(n),
					top[0]->gpu_data() + top[0]->offset(n),
					height_, width_, channels_, kernel_h_, kernel_w_,
					pad_h_, pad_w_, stride_h_, stride_w_, height_out_, width_out_,
					bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NeighborDistLayer);
}