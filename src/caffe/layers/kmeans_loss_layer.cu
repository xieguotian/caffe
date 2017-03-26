#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/kmeans_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void max_along_channel(const int nthreads, const int channels,
	const Dtype* distance, Dtype *pos, Dtype* max_value, const Dtype* label=NULL)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype maxval = -FLT_MAX;
		const Dtype* dist_data = distance + index*channels;
		if (label == NULL)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				if (maxval < dist_data[ch])
				{
					maxval = dist_data[ch];
					pos[index] = ch;
				}
			}
			max_value[index] = maxval;
		}
		else
		{
			pos[index] = label[index];
			max_value[index] = dist_data[(int)label[index]];
		}
	}
}
template <typename Dtype>
void KmeansLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//cluster_centroid_dist_layer->Forward(distance_bottom_vec_, distance_top_vec_);
	const Dtype* distance_data = bottom[0]->gpu_data(); //distance_.gpu_data();
	const int nthreads = bottom[0]->num();
	Dtype* label = NULL;
	if (bottom.size() > 1)
		label = bottom[1]->mutable_gpu_data();

	max_along_channel<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
		CAFFE_CUDA_NUM_THREADS >> >(nthreads, bottom[0]->channels(),
		distance_data,
		pos_.mutable_gpu_data(),
		max_value_set_.mutable_gpu_data(),
		label
		);

	Dtype loss = max_value_set_.asum_data();
	top[0]->mutable_cpu_data()[0] = loss / nthreads;
}

template <typename Dtype>
__global__ void kmeans_diff_bp(const int nthreads, const int channels,
	Dtype *distance_diff, const Dtype *pos, Dtype loss_weight)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int ch = pos[index];
		distance_diff[index*channels + ch] = loss_weight;
	}
}

template <typename Dtype>
void KmeansLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int nthreads = bottom[0]->num();
	kmeans_diff_bp<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
		CAFFE_CUDA_NUM_THREADS >> >(nthreads, bottom[0]->channels(),
		//distance_.mutable_gpu_diff(),
		bottom[0]->mutable_gpu_diff(),
		pos_.gpu_data(),
		-top[0]->cpu_diff()[0]/nthreads);
	//cluster_centroid_dist_layer->Backward(distance_top_vec_, propagate_down, distance_bottom_vec_);
}

INSTANTIATE_LAYER_GPU_FUNCS(KmeansLossLayer);

}  // namespace caffe
