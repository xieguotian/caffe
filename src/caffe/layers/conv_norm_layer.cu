#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
	void ConvNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		conv_layer->Forward(bottom, conv_top_vec);
		for (int n = 0; n < conv_top_vec[0]->num(); n++)
		{
			caffe_gpu_div(norm_top.count(), conv_top_vec[0]->gpu_data() + conv_top_vec[0]->offset(n),
				norm_top.gpu_data(), top[0]->mutable_gpu_data() + top[0]->offset(n));
		}
	}

	template <typename Dtype>
	void ConvNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		caffe_set(conv_top_vec[0]->count(), (Dtype)0, conv_top_vec[0]->mutable_gpu_diff());
		for (int n = 0; n < conv_top_vec[0]->num(); n++)
		{
			caffe_gpu_div(norm_top.count(), top[0]->gpu_diff() + top[0]->offset(n),
				norm_top.gpu_data(), conv_top_vec[0]->mutable_gpu_diff() + conv_top_vec[0]->offset(n));
		}

		conv_layer->Backward(conv_top_vec, propagate_down, bottom);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvNormLayer);
}