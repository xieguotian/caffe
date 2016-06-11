#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/binary_bounding_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template<typename Dtype>
	__global__ void scale_condition(const int n, Dtype alpha, Dtype beta, const Dtype* input, Dtype* output,Dtype threshold)
	{
		CUDA_KERNEL_LOOP(index, n) {
			if (input[index] >= threshold)
				output[index] = -beta;
			else
				output[index] = alpha;
		}
	} 

	template<typename Dtype>
	void BinaryBoundingLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int count = 0;
		for (int i = 0; i < bottom[1]->count(); ++i)
		{
			if (bottom[1]->cpu_data()[i] >= threshold)
				count++;
		}

		Dtype tmp_alpha = (bottom[1]->count() - count) > 0 ? alpha / (bottom[1]->count() - count) : 0;
		Dtype tmp_beta = count>0 ? beta / count: 0; 
		scale_condition<Dtype> << <CAFFE_GET_BLOCKS(bottom[1]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			bottom[1]->count(),
			tmp_alpha,
			tmp_beta,
			bottom[1]->gpu_data(),
			scalar_cache_.mutable_gpu_data(),
			threshold
			);

		if (not_initialed)
		{
			num_mul_kernel<Dtype> << <CAFFE_GET_BLOCKS(cache_tmp_.count()), CAFFE_CUDA_NUM_THREADS >> >(
				cache_tmp_.count(),
				cache_tmp_.channels(),
				1,
				bottom[0]->gpu_data(),
				scalar_cache_.gpu_data(),
				cache_tmp_.mutable_gpu_data()
				);
			caffe_gpu_gemv<Dtype>(
				CblasTrans,
				cache_tmp_.num(),
				cache_tmp_.channels(),
				(Dtype)1.0,
				cache_tmp_.gpu_data(), 
				ones_column.gpu_data(), 
				(Dtype)0.0, 
				this->blobs_[0]->mutable_gpu_data());
			LOG(INFO) << "initialed centroid.";
			not_initialed = false;
		}
		channel_sub_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			bottom[0]->count(),
			bottom[0]->channels(),
			1.0,
			bottom[0]->gpu_data(),
			this->blobs_[0]->gpu_data(),
			cache_tmp_.mutable_gpu_data());

		caffe_gpu_mul<Dtype>( 
			cache_tmp_.count(), 
			cache_tmp_.gpu_data(), 
			cache_tmp_.gpu_data(), 
			square_cache_tmp_.mutable_gpu_data());

		num_mul_kernel<Dtype> << <CAFFE_GET_BLOCKS(square_cache_tmp_.count()), CAFFE_CUDA_NUM_THREADS >> >(
			square_cache_tmp_.count(),
			square_cache_tmp_.channels(),
			1.0,
			square_cache_tmp_.gpu_data(),
			scalar_cache_.gpu_data(),
			square_cache_tmp_.mutable_gpu_data()
			);


		Dtype dot;
		//caffe_gpu_dot(
		//	square_cache_tmp_.count(),
		//	square_cache_tmp_.gpu_data(),
		//	square_cache_tmp_.mutable_gpu_data(),
		//	&dot); 
		//caffe_gpu_asum<Dtype>(square_cache_tmp_.count(), square_cache_tmp_.gpu_data(), &dot);
		//caffe_gpu_gemv<Dtype>(CblasNoTrans,
		//	1,
		//	square_cache_tmp_.count(),
		//	(Dtype)1.0/2.0,
		//	square_cache_tmp_.gpu_data(),
		//	ones_.gpu_data(),
		//	(Dtype)0.0,
		//	top[0]->mutable_gpu_data()
		//	);
		//Dtype sum = 0;
		//for (int i = 0; i < square_cache_tmp_.count(); ++i)
		//{
		//	sum += square_cache_tmp_.cpu_data()[i];
		//}
		//top[0]->mutable_cpu_data()[0] = sum / 2.0;
		caffe_gpu_dot<Dtype>(
			square_cache_tmp_.count(), 
			square_cache_tmp_.gpu_data(), 
			ones_.gpu_data(), 
			&dot);
		top[0]->mutable_cpu_data()[0] = dot / 2.0;
		//Dtype loss = dot / Dtype(2);
		//LOG(INFO) <<count<<"," <<tmp_alpha<<","<< tmp_beta<<","<< cache_tmp_.cpu_data()[0] << "," << square_cache_tmp_.cpu_data()[0] << "," << scalar_cache_.cpu_data()[0];
	    //top[0]->mutable_cpu_data()[0] /= Dtype(2);

	}

	template<typename Dtype>
	void BinaryBoundingLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		num_mul_kernel<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			bottom[0]->count(),
			bottom[0]->channels(),
			1.0,
			cache_tmp_.gpu_data(),
			scalar_cache_.gpu_data(),
			bottom[0]->mutable_gpu_diff()
			);
		caffe_gpu_scal<Dtype>(bottom[0]->count(), top[0]->cpu_diff()[0], bottom[0]->mutable_gpu_diff());
		if (update_centroid)
		{
			num_mul_kernel<Dtype> << <CAFFE_GET_BLOCKS(cache_tmp_.count()), CAFFE_CUDA_NUM_THREADS >> >(
				cache_tmp_.count(),
				cache_tmp_.channels(),
				1,
				bottom[0]->gpu_data(),
				scalar_cache_.gpu_data(),
				cache_tmp_.mutable_gpu_data()
				);
			caffe_gpu_gemv<Dtype>(
				CblasTrans, 
				cache_tmp_.num(),
				cache_tmp_.channels(),
				(Dtype)1.0-ratio,
				cache_tmp_.gpu_data(), 
				ones_column.gpu_data(), 
				ratio, 
				this->blobs_[0]->mutable_gpu_data());
		}
	}
	INSTANTIATE_LAYER_GPU_FUNCS(BinaryBoundingLossLayer);
}
