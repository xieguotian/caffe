#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
	void AmplitudeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{

	}

	template <typename Dtype>
	void AmplitudeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
		square.ReshapeLike(*bottom[0]);
		const_one.Reshape(1, bottom[0]->channels(),1,1);
		caffe_set(const_one.count(), (Dtype)(1.0), const_one.mutable_cpu_data());
	}

	template <typename Dtype>
	void AmplitudeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		int count = bottom[0]->count();
		caffe_mul(count, bottom[0]->cpu_data(), bottom[0]->cpu_data(), square.mutable_cpu_data());
		
		const Dtype* square_data = square.cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int n = 0; n < square.num(); ++n)
		{
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, square.height()*square.width(),
				square.channels(), (Dtype)1, const_one.cpu_data(), square_data + square.offset(n),
				(Dtype)0, top_data + top[0]->offset(n));
		}
		caffe_powx<Dtype>(top[0]->count(), top_data, (Dtype)1.0 / 2.0, top_data);
	}

	template <typename Dtype>
	void AmplitudeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			int count = bottom[0]->count();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* top_data = top[0]->cpu_data();
			Dtype* square_data = square.mutable_cpu_data();
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->channels(),
					bottom[0]->height()*bottom[0]->width(), 1, (Dtype)1,
					const_one.cpu_data(), top_diff + top[0]->offset(n),
					(Dtype)0, bottom_diff + bottom[0]->offset(n));

				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, square.channels(),
					square.height()*square.width(), 1, (Dtype)1,
					const_one.cpu_data(), top_data + top[0]->offset(n),
					(Dtype)0, square_data + square.offset(n));
			}

			caffe_div(count, bottom_diff, square_data, bottom_diff);
			caffe_mul(bottom[0]->count(), bottom_diff, bottom[0]->cpu_data(), bottom_diff);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(AmplitudeLayer);
#endif

	INSTANTIATE_CLASS(AmplitudeLayer);
	REGISTER_LAYER_CLASS(Amplitude);
}