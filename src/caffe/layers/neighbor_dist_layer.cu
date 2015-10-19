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
		Dtype max_num = std::numeric_limits<Dtype>::max();

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
	void NeighborDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NeighborDistLayer);
}