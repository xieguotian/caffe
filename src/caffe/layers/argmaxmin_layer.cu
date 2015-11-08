#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe
{
	template <typename Dtype>
	__global__ void ArgMaxMin(const int n, const Dtype* bottom_data,
		const int num, const int channels, const int height, const int width,
		bool is_max, Dtype* top_data,Dtype* top_filter=NULL)
	{
		CUDA_KERNEL_LOOP(index, n) {
			const int w_idx = index % width;
			const int h_idx = (index / width) % height;
			const int n_idx = (index / width / height) % num;

			const Dtype* bottom_ptr = bottom_data +
				(n_idx*channels*height + h_idx)*width + w_idx;
			int tmp_offset = height*width;

			if (is_max)
			{
				Dtype max_val = -FLT_MAX;
				top_data[index] = -1;
				for (int ch = 0; ch < channels; ++ch)
				{
					if (bottom_ptr[0] > max_val)
					{
						top_data[index] = ch;
						max_val = bottom_ptr[0];
					}

					bottom_ptr += tmp_offset;
				}
			}
			else
			{
				Dtype min_val = FLT_MAX;
				top_data[index] = -1;
				for (int ch = 0; ch < channels; ++ch)
				{
					if (bottom_ptr[0] < min_val)
					{
						top_data[index] = ch;
						min_val = bottom_ptr[0];
					}

					bottom_ptr += tmp_offset;
				}
			}
			if (top_filter != NULL)
			{
				int idx_filter = ((n_idx*channels + top_data[index])*height + h_idx)*width + w_idx;
				top_filter[idx_filter] = 1;
			}
				
		}
	}
	template <typename Dtype>
	void ArgMaxMinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int count = top[0]->count();
		Dtype* top_filter = NULL;
		if (top.size() == 2)
		{
			caffe_gpu_set(top[1]->count(), (Dtype)0, top[1]->mutable_gpu_data());
			top_filter = top[1]->mutable_gpu_data();
		}
		ArgMaxMin<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, bottom[0]->gpu_data(), bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width(), is_max_, 
			top[0]->mutable_gpu_data(),top_filter);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ArgMaxMinLayer);
}
