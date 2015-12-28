#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/shrinkage_layer.hpp"

namespace caffe
{
	template <typename Dtype>
	__global__ void ShrinkageForward(const int n, const Dtype* in, Dtype* out,
		const Dtype* threshold,int height,int width,int channels)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int h_index = index / width;
			int ch_index = h_index / height;
			int ch = (ch_index) % channels;
			if (in[index] > threshold[ch])
				out[index] = in[index] - threshold[ch];
			else if (in[index] < -threshold[ch])
				out[index] = in[index] + threshold[ch];
			else
				out[index] = 0;
		}
	}

	template <typename Dtype>
	void ShrinkageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const int count = bottom[0]->count();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* threshold = this->blobs_[0]->gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		ShrinkageForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>(
			count, bottom_data, top_data, threshold,
			bottom[0]->height(), bottom[0]->width(), bottom[0]->channels());
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void ShrinkageBackward(const int n, const Dtype*in_data, const Dtype* in_diff,
		Dtype* out_diff, const Dtype* threshold, Dtype* sign_x_data, int height, int width, int channels,int num)
	{
		CUDA_KERNEL_LOOP(index, n) {
			int w = index % width;
			int h_index = index / width;
			int h = (h_index) % height;
			int ch_index = h_index / height;
			int ch = (ch_index) % channels;
			int nn_index = ch_index / channels;
			int nn = (nn_index) % num;

			int idx = ((ch*num + nn)*height + h)*width + w;

			if (in_data[index] > threshold[ch])
			{
				out_diff[index] = in_diff[index];
				sign_x_data[idx] = -in_diff[index];
			}
			else if (in_data[index] < -threshold[ch])
			{
				out_diff[index] = in_diff[index];
				sign_x_data[idx] = in_diff[index];
			}
			else
			{
				out_diff[index] = 0;
				sign_x_data[idx] = 0;
			}
		}
	}
	template <typename Dtype>
	void ShrinkageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			const int count = bottom[0]->count();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const Dtype* threshold = this->blobs_[0]->gpu_data();
			const Dtype* top_diff = top[0]->gpu_diff();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			caffe_gpu_set(sign_x.count(), (Dtype)0, sign_x.mutable_gpu_data());

			ShrinkageBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>(
				count, bottom_data, top_diff, bottom_diff, threshold,
				sign_x.mutable_gpu_data(), bottom[0]->height(),
				bottom[0]->width(), bottom[0]->channels(), bottom[0]->num());

			const Dtype* sign_x_data = sign_x.gpu_data();
			//Dtype* threshold_diff = this->blobs_[0]->mutable_cpu_diff();
			Dtype* threshold_diff = this->blobs_[0]->mutable_gpu_diff();

			caffe_gpu_gemm(CblasNoTrans, CblasTrans, 1, sign_x.num(), sign_x.offset(1), (Dtype)1,
				ones.gpu_data(), sign_x_data, (Dtype)0, threshold_diff);
			//for (int ch = 0; ch < sign_x.num(); ++ch)
			//{
			//	caffe_gpu_asum(sign_x.offset(1), sign_x_data + sign_x.offset(ch), threshold_diff + ch);
			//}

			CUDA_POST_KERNEL_CHECK;
		}

	}

	INSTANTIATE_LAYER_GPU_FUNCS(ShrinkageLayer);
}