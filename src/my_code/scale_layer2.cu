#include <vector>
#include "caffe/Filler.hpp"
#include "caffe/layers/scale_layer2.hpp"
#include "caffe/Blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template<typename Dtype>
	void Scale2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		caffe_gpu_set<Dtype>(top[0]->count(), (Dtype)-1, top_data);

		const Dtype* scale_coeff = this->blobs_[0]->cpu_data();
		const Dtype* idx_coeff = this->blobs_[1]->cpu_data();
		//caffe_copy<Dtype>(bottom[0]->count(), bottom_data, top_data);

		int spat_dim = bottom[0]->count(2);

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				if (idx_coeff[ch] >= 0)
				{
					/*caffe_copy<Dtype>(spat_dim,
					bottom_data + (int)idx_coeff[ch] * spat_dim, top_data);

					caffe_gpu_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
					top_data);

					top_data += spat_dim;*/
					caffe_copy<Dtype>(spat_dim, bottom_data,
						top_data + (int)idx_coeff[ch] * spat_dim);
					caffe_gpu_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
						top_data + (int)idx_coeff[ch] * spat_dim);
				}
				bottom_data += spat_dim;
			}
			top_data += top[0]->count(1);
			//bottom_data += bottom[0]->count(1);
		}
	}

	template<typename Dtype>
	void Scale2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		const Dtype* scale_coeff = this->blobs_[0]->cpu_data();
		const Dtype* idx_coeff = this->blobs_[1]->cpu_data();

		int spat_dim = bottom[0]->count(2);
		//caffe_copy<Dtype>(bottom[0]->count(), top_diff, bottom_diff);

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				if (idx_coeff[ch] >= 0)
				{
					/*caffe_copy<Dtype>(spat_dim, top_diff,
					bottom_diff + (int)idx_coeff[ch] * spat_dim);

					caffe_gpu_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
					bottom_diff + (int)idx_coeff[ch] * spat_dim);

					top_diff += spat_dim;*/
					caffe_copy<Dtype>(spat_dim, top_diff + (int)idx_coeff[ch] * spat_dim,
						bottom_diff);
					caffe_gpu_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch], bottom_diff);
				}
				bottom_diff += spat_dim;
			}
			top_diff += top[0]->count(1);
			//bottom_diff += bottom[0]->count(1);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(Scale2Layer);
}