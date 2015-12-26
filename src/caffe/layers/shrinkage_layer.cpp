#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/filler.hpp"
namespace caffe
{
	template <typename Dtype>
	void ShrinkageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		ShrinkageParameter shrinkage_param = this->layer_param_.shrinkage_param();
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(1, bottom[0]->channels(), 1, 1));
		shared_ptr<Filler<Dtype>> th_filler(GetFiller<Dtype>(
			shrinkage_param.threshold_filler()));
		th_filler->Fill(this->blobs_[0].get());
	}

	template <typename Dtype>
	void ShrinkageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		top[0]->ReshapeLike(*bottom[0]);

		sign_x.Reshape(bottom[0]->channels(), bottom[0]->num(),
			bottom[0]->height(), bottom[0]->width());
		ones.Reshape(1, bottom[0]->num(), bottom[0]->height(), bottom[0]->width());
		caffe_set(ones.count(), (Dtype)1, ones.mutable_cpu_data());
	}

	template <typename Dtype>
	void ShrinkageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* threshold = this->blobs_[0]->cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int tmp_offset = bottom[0]->offset(0, 1);
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				for (int i = 0; i < tmp_offset; ++i)
				{
					if (bottom_data[i]>threshold[ch])
						top_data[i] = bottom_data[i] - threshold[ch];
					else if (bottom_data[i] < -threshold[ch])
						top_data[i] = bottom_data[i] + threshold[ch];
					else
						top_data[i] = 0;
				}
				bottom_data += tmp_offset;
				top_data += tmp_offset;
			}
		}
	}

	template <typename Dtype>
	void ShrinkageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (propagate_down[0])
		{
			const Dtype* threshold = this->blobs_[0]->cpu_data();
			Dtype* threshold_diff = this->blobs_[0]->mutable_cpu_diff();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const Dtype* top_diff = top[0]->cpu_diff();

			int tmp_offset = bottom[0]->offset(0, 1);
			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				for (int ch = 0; ch < bottom[0]->channels(); ++ch)
				{
					for (int i = 0; i < tmp_offset; ++i)
					{
						if (bottom_data[i]>threshold[ch])
						{
							bottom_diff[i] = top_diff[i];
							threshold_diff[ch] -= top_diff[i];
						}
						else if (bottom_data[i] < -threshold[ch])
						{
							bottom_diff[i] = top_diff[i];
							threshold_diff[ch] += top_diff[i];
						}

						
					}
					bottom_data += tmp_offset;
					top_diff += tmp_offset;
					bottom_diff += tmp_offset;
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(ShrinkageLayer);
#endif

	INSTANTIATE_CLASS(ShrinkageLayer);
	REGISTER_LAYER_CLASS(Shrinkage);
}