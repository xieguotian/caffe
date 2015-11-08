#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {
	template <typename Dtype>
	void ArgMaxMinLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		is_max_ = this->layer_param_.argmax_param().out_max_val();
	}

	template <typename Dtype>
	void ArgMaxMinLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
		if (top.size() == 2)
		{
			top[1]->ReshapeLike(*bottom[0]);
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(ArgMaxMinLayer);
#endif

	INSTANTIATE_CLASS(ArgMaxMinLayer);
	REGISTER_LAYER_CLASS(ArgMaxMin);
}