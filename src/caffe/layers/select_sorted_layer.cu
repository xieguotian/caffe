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
	__global__ void init_key_kernel(const int n, Dtype* key_data, Dtype* index_data, Dtype* dist_data,
		const int num,const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels ,Dtype fill_data)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w_out = index % width_col;
			int h_index = index / width_col;
			int h_out = h_index % height_col;

			int w_in_index = h_index / height_col;
			int w_in = w_in_index % kernel_w;
			int h_in_index = w_in_index / kernel_w;
			int h_in = h_in_index % kernel_h;

			//h_in += h_out * stride_h - pad_h;
			//w_in += w_out* stride_w - pad_w;

			key_data[index] = h_out*width_col + w_out;
			index_data[index] = h_in*kernel_w + w_in;
			if (dist_data[index] < 0)
				dist_data[index] = fill_data;
			//if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
			//{
			//	index_data[index] = h_in*width + w_in;
			//}
			//else
			//{
			//	index_data[index] = -1;
			//}
		}
	}

	template <typename Dtype>
	__global__ void get_top_N_index_kernel(const int n, const Dtype* index_data,
		const int num, const int top_N, const int height, const int width, const int idx_channels, Dtype* out_data)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w = index % width;
			int h_index = index / width;
			int h = h_index % height;
			int ch_index = h_index / height;
			//int ch = ch_index % channels;
			int ch = ch_index % top_N;
			//int n = ch_index / channels % num;
			int n_idx = ch_index / top_N % num;

			int idx = ((n_idx*height + h)*width + w)*idx_channels + ch;
			out_data[index] = index_data[idx];
		}
	}

	template <typename Dtype>
	__global__ void get_top_N_data_kernel(const int n, const Dtype* in_data, const Dtype* index_data,
		const int num, const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels,const int top_N, Dtype* out_data)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w_col = index % width_col;
			int h_index = index / width_col;
			int h_col = h_index % height_col;
			int top_N_index = h_index / height_col;
			int top_N_idx = top_N_index % top_N;

			int ch_index = top_N_index / top_N;
			int ch_col = ch_index % channels;
			
			

			//int n = ch_index / top_N % num;
			int n_idx = ch_index / channels % num;

			int idx = ((n_idx*top_N + top_N_idx)*height_col + h_col)*width_col + w_col;
			int d_idx = index_data[idx];
			
			//============
			int h_in = h_col * stride_h - pad_h + (int)(d_idx / kernel_w%kernel_h);
			int w_in = w_col* stride_w - pad_w + (int)(d_idx%kernel_w);
			//d_idx = h_in*width + w_in;
			//==========
			//if (d_idx >= 0)
			if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
			{
				//int w_in = d_idx % width;
				//int h_in = (d_idx / width) % height;
				int data_index = ((n_idx*channels + ch_col)*height + h_in)*width + w_in;
				out_data[index] = in_data[data_index];
			}
			else
				out_data[index] = 0;
		}
	}

	template <typename Dtype>
	void SelectSortedLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		nei_dist_layer->Forward(nei_dist_bottom_vec, nei_dist_top_vec);
		thrust::device_ptr<Dtype> dist_ptr(dist.mutable_gpu_data());
		thrust::device_ptr<Dtype> key_ptr(key.mutable_gpu_data());
		thrust::device_ptr<Dtype> index_ptr(index.mutable_gpu_data());

		//initial key and index
		int num_kernels = key.count();
		init_key_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
			num_kernels, key.mutable_gpu_data(), index.mutable_gpu_data(), dist.mutable_gpu_data(),
			num_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
			height_out_, width_out_, channels_, std::numeric_limits<Dtype>::max());
		CUDA_POST_KERNEL_CHECK;

		//sort distance
		int count = dist.offset(1);
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			thrust::zip_iterator<thrust::tuple<thrust::device_ptr<Dtype>, 
				thrust::device_ptr<Dtype>>> key_index_tuple = thrust::make_zip_iterator(
				thrust::make_tuple(key_ptr + key.offset(n), index_ptr + index.offset(n)));
			thrust::zip_iterator<thrust::tuple<thrust::device_ptr<Dtype>,
				thrust::device_ptr<Dtype>>> dist_index_tuple = thrust::make_zip_iterator(
				thrust::make_tuple(dist_ptr+dist.offset(n), index_ptr+index.offset(n)));

			thrust::stable_sort_by_key(dist_ptr + dist.offset(n),
				dist_ptr + dist.offset(n) + count, key_index_tuple);
			thrust::stable_sort_by_key(key_ptr + key.offset(n), 
				key_ptr + key.offset(n) + count, dist_index_tuple);
		}

		//get top N index
		num_kernels = top[0]->count();
		get_top_N_index_kernel<Dtype> << < CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
			num_kernels, index.gpu_data(), top[0]->num(), 
			top[0]->channels(), top[0]->height(), top[0]->width(),
			index.channels(),top[0]->mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;

		// get top N data
		if (top.size() == 2)
		{
			num_kernels = top[1]->count();
			get_top_N_data_kernel<Dtype> << < CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
				num_kernels, bottom[1]->gpu_data(), top[0]->gpu_data(),
				num_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				height_out_, width_out_, channels_, top_N_, top[1]->mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
		}
	}

	template <typename Dtype>
	__global__ void top_N_diff_backward_kernel(const int n, const Dtype* in_diff, const Dtype* index_data,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		const int height_col, const int width_col, const int channels, const int top_N, Dtype* out_diff)
	{
		CUDA_KERNEL_LOOP(index, n) {
			Dtype val = 0;
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
						//=========
						int h_in = h_col * stride_h - pad_h + (int)(d_idx_data / kernel_w%kernel_h);
						int w_in = w_col* stride_w - pad_w + (int)(d_idx_data%kernel_w);
						if (h_in >= 0 && w_in >= 0 && h_in < height&&w_in < width)
						{
							d_idx_data = h_in*width + w_in;
						}
						else
						{
							d_idx_data = -1;
						}
						//=========
						if (d_idx_data == d_idx)
						{
							int in_diff_idx = ((c*top_N + tn)*height_col + h_col)*width_col + w_col;
							val += in_diff[in_diff_idx];
						}
					}
				}
			}
			out_diff[index] = val;
		}
	}

	template <typename Dtype>
	void SelectSortedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
		}

		if (top.size()==2 && propagate_down[1])
		{
			int num_kernels = channels_*height_*width_;
			for (int n = 0; n < top[1]->num(); ++n)
			{
				top_N_diff_backward_kernel<Dtype> << < CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS >> >(
					num_kernels, top[1]->gpu_diff() + top[1]->offset(n), top[0]->gpu_data() + top[0]->offset(n),
					height_, width_, kernel_h_, kernel_w_,
					pad_h_, pad_w_, stride_h_, stride_w_,
					height_out_, width_out_, channels_, top_N_,
					bottom[1]->mutable_gpu_diff() + bottom[1]->offset(n));
				CUDA_POST_KERNEL_CHECK;
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SelectSortedLayer);
}