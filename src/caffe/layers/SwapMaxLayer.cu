#include "caffe/layers/SwapMaxLayer.hpp"

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include <limits.h>

namespace caffe{
	
	template<typename Dtype>
	__global__ void SwapMaxKernel(const int count,const int channel, const int height, const int width,
		const Dtype min_val, const Dtype* label_data, Dtype* data)
	{
		CUDA_KERNEL_LOOP(index, count) {
			int w = index % width;
			int res_w = index / width;
			int h = res_w % height;
			int n = res_w / height;

			int spat_dim = height*width;
			Dtype* st_idx = data + (n*channel*height + h)*width + w;
			
			// sel max value
			Dtype max_pred = min_val;
			int pos = -1;

			int label = label_data[index];
			for (int idx = 0; idx < channel; ++idx)
			{
				if (st_idx[idx*spat_dim]>max_pred)
				{
					pos = idx;
					max_pred = st_idx[idx*spat_dim];
				}
			}

			// swap value between label and max value.
			Dtype tmp = st_idx[label*spat_dim];
			st_idx[label*spat_dim] = st_idx[pos*spat_dim];
			st_idx[pos*spat_dim] = tmp;
		}
	}
	template<typename Dtype>
	void SwapMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* label_data = bottom[1]->gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype max_pred = std::numeric_limits<Dtype>::lowest();

		int threads = bottom[0]->num()*bottom[0]->height()*bottom[1]->width();
		caffe_copy(top[0]->count(), bottom_data, top_data);
		SwapMaxKernel<Dtype> << <CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS >> >(
			threads,
			bottom[0]->channels(), 
			bottom[0]->height(), 
			bottom[0]->width(), 
			max_pred,
			label_data, 
			top_data
			);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SwapMaxLayer);
}