#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe
{
	template <typename Dtype>
	__global__ void NormalizeForward(const int n, const Dtype* in, Dtype* out, 
		Dtype* norm_cache_data, int height, int width, int channels,int num,Dtype eps)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w = index % width;
			int h = (index / width) % height;
			int nn = (index / width / height) % num;

			for (int ch = 0; ch < channels; ++ch)
			{
				int idx = ((nn*channels + ch)*height + h)*width + w;
				norm_cache_data[index] += in[idx];
			}
			norm_cache_data[index] += eps;
			for (int ch = 0; ch < channels; ++ch)
			{
				int idx = ((nn*channels + ch)*height + h)*width + w;
				out[idx] = in[idx] / norm_cache_data[index];
			}
		}
	}

	template <typename Dtype>
	void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		/*
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* norm_cache_data = norm_cache_.mutable_gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		
		int count = norm_cache_.count();
		caffe_gpu_set(count, (Dtype)0, norm_cache_data);
		const Dtype eps = (Dtype)std::numeric_limits<Dtype>::epsilon();
		NormalizeForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, bottom_data, top_data,norm_cache_data, bottom[0]->height(), bottom[0]->width(),
			bottom[0]->channels(), bottom[0]->num(),eps);


		CUDA_POST_KERNEL_CHECK;
		*/

		Dtype* norm_cache_data = norm_cache_.mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		caffe_gpu_set(norm_cache_.count(), (Dtype)0, norm_cache_data);
		int tmp_offset = norm_cache_.offset(1);
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				caffe_gpu_axpy(tmp_offset, (Dtype)1,
					bottom_data, norm_cache_data);
				bottom_data += tmp_offset;
			}
			norm_cache_data += tmp_offset;
		}
		caffe_gpu_add_scalar(norm_cache_.count(), (Dtype)std::numeric_limits<Dtype>::epsilon(), norm_cache_data);

		norm_cache_data = norm_cache_.mutable_gpu_data();
		bottom_data = bottom[0]->gpu_data();

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				caffe_gpu_div(tmp_offset, bottom_data, norm_cache_data, top_data);
				bottom_data += tmp_offset;
				top_data += tmp_offset;
			}
			norm_cache_data += tmp_offset;
		}
	}

	template <typename Dtype>
	__global__ void NormalizeBackward(const int n, const Dtype* in_data, Dtype* out_diff,const Dtype* in_diff,
		const Dtype* norm_cache_data, int height, int width, int channels, int num)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w = index % width;
			int h = (index / width) % height;
			int nn = (index / width / height) % num;
			Dtype val = 0;
			for (int ch = 0; ch < channels; ++ch)
			{
				int idx = ((nn*channels + ch)*height + h)*width + w;
				val += in_diff[idx] * in_data[idx] / norm_cache_data[index] / norm_cache_data[index];
			}
			for (int ch = 0; ch < channels; ++ch)
			{
				int idx = ((nn*channels + ch)*height + h)*width + w;
				out_diff[idx] = in_diff[idx] / norm_cache_data[index] - val;
			}
		}
	}
	template <typename Dtype>
	void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		/*
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* norm_cache_data = norm_cache_.gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		//int count = bottom[0]->count();
		int count = norm_cache_.count();
		NormalizeBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, bottom_data, bottom_diff, top_diff, norm_cache_data, bottom[0]->height(),
			bottom[0]->width(), bottom[0]->channels(), bottom[0]->num());
		*/
		Dtype* norm_cache_diff = norm_cache_.mutable_gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* norm_cache_data = norm_cache_.gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();

		caffe_gpu_mul(bottom[0]->count(), top_diff, bottom_data, bottom_diff);
		caffe_gpu_set(norm_cache_.count(), (Dtype)0, norm_cache_diff);
		int tmp_offset = norm_cache_.offset(1);
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				caffe_gpu_div(tmp_offset, bottom_diff, norm_cache_data, bottom_diff);
				caffe_gpu_div(tmp_offset, bottom_diff, norm_cache_data, bottom_diff);
				caffe_gpu_axpy(tmp_offset, (Dtype)1, bottom_diff, norm_cache_diff);
				bottom_diff += tmp_offset;
			}
			norm_cache_data += tmp_offset;
			norm_cache_diff += tmp_offset;
		}

		norm_cache_diff = norm_cache_.mutable_gpu_diff();
		norm_cache_data = norm_cache_.gpu_data();
		bottom_diff = bottom[0]->mutable_gpu_diff();
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				caffe_gpu_div(tmp_offset, top_diff, norm_cache_data, bottom_diff);
				caffe_gpu_sub(tmp_offset, bottom_diff, norm_cache_diff, bottom_diff);
				bottom_diff += tmp_offset;
				top_diff += tmp_offset;
			}
			norm_cache_data += tmp_offset;
			norm_cache_diff += tmp_offset;
		}

	}

	INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);
}