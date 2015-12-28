#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/euclidean_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void EuclideanLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{

	}

	template <typename Dtype>
	void EuclideanLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
			<< "Inputs must have the same dimension.";
		top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
		diff.ReshapeLike(*bottom[0]);
		square.ReshapeLike(*bottom[0]);
		const_one.Reshape(1, bottom[0]->channels(),1,1);
		caffe_set(const_one.count(), (Dtype)(1.0), const_one.mutable_cpu_data());
	}

	template <typename Dtype>
	void EuclideanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int count = bottom[0]->count();
		caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),diff.mutable_cpu_data());
		caffe_mul(count, diff.cpu_data(), diff.cpu_data(), square.mutable_cpu_data());
		
		const Dtype* square_data = square.cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int n = 0; n < square.num(); ++n)
		{
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, square.height()*square.width(),
				square.channels(), (Dtype)0.5, const_one.cpu_data(), square_data + square.offset(n),
				(Dtype)0, top_data + top[0]->offset(n));
		}
	}

	template <typename Dtype>
	void EuclideanLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Dtype* top_diff = top[0]->mutable_cpu_diff();

		for (int i = 0; i < 2; i++)
		{
			if (propagate_down[i])
			{
				const Dtype sign = (i == 0) ? 1 : -1;
				Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
				for (int n = 0; n < diff.num(); ++n)
				{
					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[i]->channels(),
						bottom[i]->height()*bottom[i]->width(), 1, (Dtype)sign,
						const_one.cpu_data(), top_diff + top[0]->offset(n),
						(Dtype)0, bottom_diff + bottom[i]->offset(n));
				}

				caffe_mul(bottom[i]->count(), bottom_diff, diff.cpu_data(), bottom_diff);
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(EuclideanLayer);
#endif

	INSTANTIATE_CLASS(EuclideanLayer);
	REGISTER_LAYER_CLASS(Euclidean);
}