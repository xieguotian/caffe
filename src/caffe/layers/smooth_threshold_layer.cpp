#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/smooth_threshold_layer.hpp"
#include "caffe/filler.hpp"

namespace caffe {
	template <typename Dtype>
	void SmoothThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		SmoothThresholdParameter smooth_param = this->layer_param_.smooth_threshold_param();
		alpha = (Dtype)smooth_param.alpha();
		beta = (Dtype)smooth_param.beta();

		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
		shared_ptr<Filler<Dtype>> th_filler(GetFiller<Dtype>(
			smooth_param.threshold_filler()));
		th_filler->Fill(this->blobs_[0].get());

	}

	template <typename Dtype>
	void SmoothThresholdLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		top[0]->ReshapeLike(*bottom[0]);
		diff.ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void SmoothThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* threshold = this->blobs_[0]->cpu_data();

		Dtype *top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();

		for (int i = 0; i < top[0]->count(); i++)
		{
			top_data[i] = 1.0 / (1.0 + alpha*exp(beta*(-abs(bottom_data[i]) + threshold[0])));
		}
	}

	template <typename Dtype>
	void SmoothThresholdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			const Dtype* top_data = top[0]->cpu_data();
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* threshold_diff = this->blobs_[0]->mutable_cpu_diff();

			const int count = bottom[0]->count();

			Dtype sigmoid_x;
			Dtype sign_x;
			for (int i = 0; i < count; ++i)
			{
				sigmoid_x = top_data[i];
				sign_x = (Dtype(0) < bottom_data[i]) - (bottom_data[i] < Dtype(0));
				bottom_diff[i] = top_diff[i] * sigmoid_x*(1.0 - sigmoid_x)*beta*sign_x;
				threshold_diff[0] += top_diff[i] * sigmoid_x*(sigmoid_x - 1.0)*beta;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(SmoothThresholdLayer);
#endif

	INSTANTIATE_CLASS(SmoothThresholdLayer);
	REGISTER_LAYER_CLASS(SmoothThreshold);
}