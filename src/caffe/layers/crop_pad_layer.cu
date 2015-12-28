#include <vector>

#include "caffe/layers/crop_pad_layer.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void CropPadLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	caffe_gpu_set(top[0]->count(), (Dtype)0, top[0]->mutable_gpu_data());

	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();

	const int width = std::min(bottom[0]->width(), top[0]->width());
	const int height = std::min(bottom[0]->height(), top[0]->height());
	const int channels = std::min(bottom[0]->channels(), top[0]->channels());
	const int num = std::min(bottom[0]->num(), top[0]->num());

	const int top_offset_n = top[0]->offset(0,1);
	const int top_offset_ch = top[0]->offset(0, 0, 1);
	const int top_offset_height = top[0]->offset(0, 0, 0, 1);

	const int bottom_offset_n = bottom[0]->offset(0,1);
	const int bottom_offset_ch = bottom[0]->offset(0, 0, 1);
	const int bottom_offset_height = bottom[0]->offset(0, 0, 0, 1);

	Dtype* top_ptr = top_data;
	const Dtype* bottom_ptr = bottom_data;
	for (int n = 0; n < num; ++n)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			top_ptr = top_data + n*top_offset_n + ch*top_offset_ch;
			bottom_ptr = bottom_data + n*bottom_offset_n + ch*bottom_offset_ch;

			for (int h = 0; h < height; ++h)
			{
				caffe_copy(width, bottom_data, top_data);
				bottom_ptr += top_offset_height;;
				top_ptr += bottom_offset_height;
			}
		}
	}
}

template <typename Dtype>
void CropPadLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	if (bottom.size() == 2 && propagate_down[1])
	{
		LOG(FATAL) << this->type()
			<< " Layer cannot backpropagate to second bottom.";
	}

	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

	const int width = std::min(bottom[0]->width(), top[0]->width());
	const int height = std::min(bottom[0]->height(), top[0]->height());
	const int channels = std::min(bottom[0]->channels(), top[0]->channels());
	const int num = std::min(bottom[0]->num(), top[0]->num());

	const int top_offset_n = top[0]->offset(0, 1);
	const int top_offset_ch = top[0]->offset(0, 0, 1);
	const int top_offset_height = top[0]->offset(0, 0, 0, 1);

	const int bottom_offset_n = bottom[0]->offset(0, 1);
	const int bottom_offset_ch = bottom[0]->offset(0, 0, 1);
	const int bottom_offset_height = bottom[0]->offset(0, 0, 0, 1);

	const Dtype* top_ptr = top_diff;
	Dtype* bottom_ptr = bottom_diff;
	for (int n = 0; n < num; ++n)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			top_ptr = top_diff + n*top_offset_n + ch*top_offset_ch;
			bottom_ptr = bottom_diff + n*bottom_offset_n + ch*bottom_offset_ch;

			for (int h = 0; h < height; ++h)
			{
				caffe_copy(width, top_diff, bottom_diff);
				bottom_ptr += top_offset_height;;
				top_ptr += bottom_offset_height;
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(CropPadLayer);

}  // namespace caffe
