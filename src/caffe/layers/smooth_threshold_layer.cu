#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe
{
	template <typename Dtype>
	__global__ void ThresholdForward(const int n, const Dtype* in, Dtype* out,
		const Dtype* threshold, const Dtype beta, const Dtype alpha) 
	{
		CUDA_KERNEL_LOOP(index, n) {
			out[index] = 1. / (1. + alpha*exp(beta*(-abs(in[index])+threshold[0])));
		}
	}

	template <typename Dtype>
	void SmoothThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const int count = bottom[0]->count();
		const Dtype* threshold = this->blobs_[0]->gpu_data();
		ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>(
			count, bottom_data, top_data, threshold, beta, alpha);

		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void ThresholdBackward(const int n, const Dtype* in_diff,
		const Dtype* out_data, const Dtype* in_data, Dtype* out_diff, Dtype* threshold_diff,
		const Dtype beta) {
		CUDA_KERNEL_LOOP(index, n) {
			const Dtype sigmoid_x = out_data[index];
			Dtype sign_x = (Dtype(0) < in_data[index]) - (in_data[index] < Dtype(0));
			out_diff[index] = in_diff[index] * sigmoid_x * (1.0 - sigmoid_x) * beta * sign_x;
			threshold_diff[index] = sigmoid_x * (sigmoid_x - 1.0) * beta;
		}
	}
	template <typename Dtype>
	void SmoothThresholdLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			const Dtype* top_data = top[0]->gpu_data();
			const Dtype* top_diff = top[0]->gpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			const int count = bottom[0]->count();
			Dtype* diff_th = diff.mutable_gpu_data();

			ThresholdBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>(
				count, top_diff, top_data, bottom_data, bottom_diff, diff_th, beta);

			CUDA_POST_KERNEL_CHECK;

			Dtype* threshold_diff = this->blobs_[0]->mutable_cpu_diff();
			caffe_gpu_dot(count, diff.gpu_data(), top[0]->gpu_diff(), threshold_diff);

		}
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SmoothThresholdLayer);
}
