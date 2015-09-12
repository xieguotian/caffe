#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe
{

	template <typename Dtype>
	void EuclideanLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int count = bottom[0]->count();
		caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff.mutable_gpu_data());
		caffe_gpu_mul(count, diff.gpu_data(), diff.gpu_data(), square.mutable_gpu_data());

		const Dtype* square_data = square.gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		for (int n = 0; n < square.num(); ++n)
		{
			caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, square.height()*square.width(),
				square.channels(), (Dtype)0.5, const_one.gpu_data(), square_data + square.offset(n),
				(Dtype)0, top_data + top[0]->offset(n));
		}
	}


	template <typename Dtype>
	void EuclideanLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Dtype* top_diff = top[0]->mutable_gpu_diff();

		for (int i = 0; i < 2; i++)
		{
			if (propagate_down[i])
			{
				const Dtype sign = (i == 0) ? 1 : -1;
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < diff.num(); ++n)
				{
					caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, bottom[i]->channels(),
						bottom[i]->height()*bottom[i]->width(), 1, (Dtype)sign,
						const_one.gpu_data(), top_diff + top[0]->offset(n),
						(Dtype)0, bottom_diff + bottom[i]->offset(n));
				}

				caffe_gpu_mul(bottom[i]->count(), bottom_diff, diff.gpu_data(), bottom_diff);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLayer);
}
