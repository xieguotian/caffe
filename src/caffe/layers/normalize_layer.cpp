#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		
	}

	template <typename Dtype>
	void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		top[0]->ReshapeLike(*bottom[0]);
		norm_cache_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
		ones_.Reshape(1, bottom[0]->channels(), 1, 1);
		caffe_set(ones_.count(), (Dtype)1, ones_.mutable_cpu_data());
	}

	template <typename Dtype>
	void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* norm_cache_data = norm_cache_.mutable_cpu_data();

		//caffe_set(norm_cache_.count(), (Dtype)0, norm_cache_data);
		
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		for (int n = 0; n < num; ++n)
		{
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, height*width, channels, (Dtype)1,
				ones_.cpu_data(), bottom_data + bottom[0]->offset(n),
				(Dtype)0, norm_cache_data + norm_cache_.offset(n));
		}

		caffe_add_scalar(norm_cache_.count(), (Dtype)std::numeric_limits<Dtype>::epsilon(), norm_cache_data);
		int tmp_offset = norm_cache_.offset(1);
		for (int n = 0; n < num; ++n)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				caffe_div(tmp_offset, bottom_data, norm_cache_data, top_data);
				//caffe_copy(tmp_offset, norm_cache_data, top_data);
				bottom_data += tmp_offset;
				top_data += tmp_offset;
			}
			norm_cache_data += tmp_offset;
		}
		
	}

	template <typename Dtype>
	void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* norm_cache_data = norm_cache_.cpu_data();

		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* norm_cache_diff = norm_cache_.mutable_cpu_diff();

		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		caffe_mul(bottom[0]->count(), top_diff, bottom_data, bottom_diff);
		for (int n = 0; n < num; ++n)
		{
			caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, height*width, channels, (Dtype)1,
				ones_.cpu_data(), bottom_diff + bottom[0]->offset(n),
				(Dtype)0, norm_cache_diff + norm_cache_.offset(n));
		}
		caffe_div(norm_cache_.count(), norm_cache_diff, norm_cache_data, norm_cache_diff);
		caffe_div(norm_cache_.count(), norm_cache_diff, norm_cache_data, norm_cache_diff);

		for (int n = 0; n < num; ++n)
		{
			for (int h = 0; h < height; ++h)
			{
				for (int w = 0; w < width; ++w)
				{
					int idx1 = (n*height + h)*width + w;
					for (int ch = 0; ch < channels; ++ch)
					{
						int idx2 = ((n*channels + ch)*height + h)*width + w;

						bottom_diff[idx2] = top_diff[idx2] / norm_cache_data[idx1] - norm_cache_diff[idx1];
					}
				}
			}
		}

	}
#ifdef CPU_ONLY
	STUB_GPU(NormalizeLayer);
#endif

	INSTANTIATE_CLASS(NormalizeLayer);
	REGISTER_LAYER_CLASS(Normalize);
}