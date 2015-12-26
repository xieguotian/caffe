#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/filter_elt_layer.hpp"

namespace caffe {

template <typename Dtype>
void FilterEltLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void FilterEltLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {



	CHECK_EQ(bottom[1]->shape(0), bottom[0]->shape(0)) <<
		"Each bottom should have the same 0th dimension as the selector blob";
	CHECK_EQ(bottom[1]->shape(1), bottom[0]->shape(1)) <<
		"Each bottom should have the same 1th dimension as the selector blob";
	CHECK_EQ(bottom[1]->shape(2), bottom[0]->shape(2)) <<
		"Each bottom should have the same 2th dimension as the selector blob";
	CHECK_EQ(bottom[1]->shape(3), bottom[0]->shape(3)) <<
		"Each bottom should have the same 3th dimension as the selector blob";


	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FilterEltLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* filter = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	for (int i = 0; i < bottom[0]->count(); ++i)
	{
		if (filter[i]>0)
			top_data[i] = bottom_data[i];
		else
			top_data[i] = 0;
	}
}

template <typename Dtype>
void FilterEltLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0])
		return;
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(FilterEltLayer);
#endif

INSTANTIATE_CLASS(FilterEltLayer);
REGISTER_LAYER_CLASS(FilterElt);

}  // namespace caffe
