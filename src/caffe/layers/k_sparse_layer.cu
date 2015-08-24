#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void ChannelSparseFroward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width, const int sparse_k,
		Dtype* const top_data, int* rank_idx, Dtype* top_mask, Dtype* rank_val)
	{
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = (index / width / height) % num;

			int stride_r = n*sparse_k*height*width + h*width + w;
			int stride_t = n*channels*height*width + h*width + w;
			for (int ch = 0; ch < channels; ch++)
			{
				int index_b = stride_t + ch*height*width;
				Dtype b_val = bottom_data[index_b];

				int cr;
				int index_r;
				int index_r_last;
				for (cr = 0; cr < sparse_k; ++cr)
				{
					index_r = stride_r + cr*height*width;
					if (rank_val[index_r] >= b_val)
						break;
					if (cr + 1 < sparse_k)
					{
						index_r_last = stride_r + (cr + 1)*height*width;
						rank_val[index_r] = rank_val[index_r_last];
						rank_idx[index_r] = rank_idx[index_r_last];
					}
				}
				if (cr - 1 >= 0)
				{
					index_r = stride_r + (cr - 1)*height*width;
					rank_val[index_r] = b_val;
					rank_idx[index_r] = ch;
				}
			}

			for (int cr = 0; cr < sparse_k; ++cr)
			{
				int index_r = stride_r + cr*height*width;
				int index_t = stride_t + rank_idx[index_r] * height*width;
				if (top_mask)
					top_mask[index_r] = (Dtype)rank_idx[index_r];
				top_data[index_t] = rank_val[index_r];
			}
		}
	}

	template <typename Dtype>
	void KSparseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		int top_count = top[0]->count();
		caffe_gpu_set(top_count, Dtype(0), top_data);
		
		int count = bottom[0]->num()*bottom[0]->height()*bottom[0]->width();
		int* rank_idx = rank_idx_.mutable_gpu_data();
		Dtype* rank_val = rank_val_.mutable_gpu_data();
		Dtype* top_mask = NULL;
		int rank_count = rank_val_.count();
		caffe_gpu_set(rank_count, Dtype(-FLT_MAX), rank_val);

		switch (this->layer_param_.k_sparse_param().sparse_type())
		{
		case KSparseParameter_SparseMethod_CHANNEL:
			if (top.size() > 1)
				top_mask = top[1]->mutable_gpu_data();

			ChannelSparseFroward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom_data, bottom[0]->num(), bottom[0]->channels(),
				bottom[0]->height(), bottom[0]->width(), rank_idx_.channels(),
				top_data, rank_idx, top_mask, rank_val);
			break;
		case KSparseParameter_SparseMethod_SPAT:
			NOT_IMPLEMENTED;
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void ChannelSparseBackward(const int nthreads,
		const Dtype* const top_diff, const int num, const int channels,
		const int height, const int width, const int sparse_k,
		Dtype* const bottom_diff, const int* rank_idx)
	{
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int w = index % width;
			const int h = (index / width) % height;
			const int c = (index / width / height) % sparse_k;
			const int n = (index / width / height / sparse_k) % num;

			int index_r = ((n*sparse_k + c)*height + h)*width + w;
			int index_b = ((n*channels + rank_idx[index_r])*height + h)*width + w;
			bottom_diff[index_b] += top_diff[index_b];
		}
	}

	template <typename Dtype>
	void KSparseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);

		int count = rank_idx_.count();
		const int* rank_idx = rank_idx_.gpu_data();
		switch (this->layer_param_.k_sparse_param().sparse_type())
		{
		case KSparseParameter_SparseMethod_CHANNEL:
			ChannelSparseBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, top_diff, top[0]->num(), top[0]->channels(),
				top[0]->height(), top[0]->width(), rank_idx_.channels(),
				bottom_diff, rank_idx);
			break;
		case KSparseParameter_SparseMethod_SPAT:
			NOT_IMPLEMENTED;
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}

		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(KSparseLayer);
}