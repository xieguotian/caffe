#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

#include "thrust\sort.h"
#include "thrust\device_vector.h"

namespace caffe 
{
	template <typename Dtype>
	__global__ void select_replace_kernel(const int n, const Dtype* in_data, const Dtype* index_data,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels, const int top_N, 
		Dtype* out_data,Dtype* count_coef,Dtype* top_1_data=NULL)
	{
		CUDA_KERNEL_LOOP(index, n) {
			Dtype val = 0;
			Dtype count = 0;
			int w = index % width + pad_w;
			int h = (index / width) % height + pad_h;
			int c = index / (width * height);

			int d_idx = index % (width*height);//h*width + w;

			// compute the start and end of the output
			int w_col_start = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
			int w_col_end = min(w / stride_w + 1, width_col);
			int h_col_start = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
			int h_col_end = min(h / stride_h + 1, height_col);

			for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
			{
				for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
				{
					for (int tn = 0; tn < top_N; ++tn)
					{
						int tn_idx = (tn*height_col + h_col)*width_col + w_col;
						int d_idx_data = index_data[tn_idx];
						//========
						//int h_in = h_col * stride_h - pad_h + (int)(d_idx_data / kernel_w%kernel_h);
						//int w_in = w_col* stride_w - pad_w + (int)(d_idx_data%kernel_w);
						//if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
						//{
						//	d_idx_data = h_in*width + w_in;
						//}
						//else
						//{
						//	d_idx_data = -1;
						//}
						//========
						if (d_idx_data == d_idx)
						{
							int in_data_idx = ((c*top_N + tn)*height_col + h_col)*width_col + w_col;
							val += in_data[in_data_idx];
							count += 1;
						}
					}
				} 
			}

			if (c == 0)
			{
				count_coef[d_idx] = (Dtype)count;
			}
			if (top_1_data != NULL)
			{
				top_1_data[index] = (count > 0) ? 1 : 0;
			}
			out_data[index] = val;
			if (count != 0)
				out_data[index] /= count;
		}
	}

	
	template <typename Dtype>
	__global__ void transfer_index_kernel(const int n, const Dtype* index_data,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels, const int top_N,
		Dtype* out_idx)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w_col = index % width_col;
			int h_index = index / width_col;
			int h_col = h_index % height_col;
			int tn_index = h_index / height_col;
			int tn = tn_index % top_N;
			int n_idx = tn_index / top_N;

			int d_idx_data = index_data[index];
			int h_in = h_col * stride_h - pad_h + (int)(d_idx_data / kernel_w%kernel_h);
			int w_in = w_col* stride_w - pad_w + (int)(d_idx_data%kernel_w);

			if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
			{
				out_idx[index] = h_in*width + w_in;
			}
			else
			{
				out_idx[index] = -1;
			}
		}
	}

	template <typename Dtype>
	void SelectReplaceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int num_kernels = channels_*height_*width_;
		caffe_gpu_set(top[0]->count(), (Dtype)0, top[0]->mutable_gpu_data());
		caffe_gpu_set(count_coef_.count(), (Dtype)0, count_coef_.mutable_gpu_data());

		transfer_index_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom[1]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			bottom[1]->count(), bottom[1]->gpu_data(),
			height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
			stride_h_, stride_w_, height_out_, width_out_, channels_, top_N_,
			idx_trans_cache_.mutable_gpu_data());


		for (int n = 0; n < num_; ++n)
		{
			Dtype* top_1_data = NULL;
			if (top.size() == 2)
				top_1_data = top[1]->mutable_gpu_data()+top[1]->offset(n);

			select_replace_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
				num_kernels,bottom[0]->gpu_data()+bottom[0]->offset(n),
				//bottom[1]->gpu_data()+bottom[1]->offset(n),
				idx_trans_cache_.gpu_data() + idx_trans_cache_.offset(n),
				height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
				stride_h_, stride_w_, height_out_, width_out_, channels_, top_N_,
				top[0]->mutable_gpu_data()+top[0]->offset(n), 
				count_coef_.mutable_gpu_data()+count_coef_.offset(n),
				top_1_data);

			CUDA_POST_KERNEL_CHECK;
		}
		
	}

	
	template <typename Dtype>
	__global__ void select_replace_backward_kernel(const int n, 
		const Dtype* in_diff, const Dtype* index_data,const Dtype* count_coef,
		const int num, const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels, const int top_N, Dtype* out_diff)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w_col = index % width_col;
			int h_index = index / width_col;
			int h_col = h_index % height_col;
			int top_N_index = h_index / height_col;
			int top_N_idx = top_N_index % top_N;

			int ch_index = top_N_index / top_N;
			int ch_col = ch_index % channels;

			int idx = (top_N_idx*height_col + h_col)*width_col + w_col;
			int d_idx = index_data[idx];
			//int count = count_coef[idx];
			//============
			//int h_in = h_col * stride_h - pad_h + (int)(d_idx / kernel_w%kernel_h);
			//int w_in = w_col* stride_w - pad_w + (int)(d_idx%kernel_w);
			//==========
			if (d_idx >= 0)
			//if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
			{
				int w_in = d_idx % width;
				int h_in = (d_idx / width) % height;
				int data_index = (ch_col*height + h_in)*width + w_in;
				int count = count_coef[h_in*width + w_in];
				out_diff[index] =  in_diff[data_index] / count;
			}
			else
			{
				out_diff[index] = 0;
			}
		}
	}

	template <typename Dtype>
	void SelectReplaceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			int num_kernels = channels_*top_N_*height_out_*width_out_;
			for (int n = 0; n < num_; ++n)
			{
				select_replace_backward_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
					num_kernels, top[0]->gpu_diff() + top[0]->offset(n),
					//bottom[1]->gpu_data() + bottom[1]->offset(n),
					idx_trans_cache_.gpu_data() + idx_trans_cache_.offset(n),
					count_coef_.gpu_data() + count_coef_.offset(n),
					num_, height_, width_, kernel_h_, kernel_w_,
					pad_h_, pad_w_, stride_h_, stride_w_,
					height_out_, width_out_, channels_, top_N_,
					bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));
				CUDA_POST_KERNEL_CHECK;
			}
		}

		if (propagate_down[1])
		{

		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SelectReplaceLayer);
}