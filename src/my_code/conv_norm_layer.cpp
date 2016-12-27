#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/conv_norm_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void ConvNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";

		conv_layer.reset(new ConvolutionLayer<Dtype>(this->layer_param()));
		conv_top_vec.push_back(&conv_top);
		conv_layer->SetUp(bottom, conv_top_vec);

		LayerParameter conv_param(this->layer_param());
		conv_param.mutable_convolution_param()->set_bias_term(false);
		conv_param.mutable_convolution_param()->mutable_weight_filler()->set_type("constant");
		conv_param.mutable_convolution_param()->mutable_weight_filler()->set_value(1);
		//conv_param.mutable_convolution_param()->set_num_output(1);

		norm_bottom.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		caffe_set(norm_bottom.count(), (Dtype)1, norm_bottom.mutable_cpu_data());
		norm_top_vec.push_back(&norm_top);
		norm_bottom_vec.push_back(&norm_bottom);
		
		norm_layer.reset(new ConvolutionLayer<Dtype>(conv_param));
		norm_layer->SetUp(norm_bottom_vec, norm_top_vec);
		norm_layer->Forward(norm_bottom_vec, norm_top_vec);

		bool bias_term = this->layer_param_.convolution_param().bias_term();
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			if (bias_term) {
				this->blobs_.resize(2);
			}
			else {
				this->blobs_.resize(1);
			}
			this->blobs_[0].reset(new Blob<Dtype>(conv_layer->blobs()[0]->shape()));
			this->blobs_[0]->ShareData(*conv_layer->blobs()[0].get());
			this->blobs_[0]->ShareDiff(*conv_layer->blobs()[0].get());

			if (bias_term)
			{
				this->blobs_[1].reset(new Blob<Dtype>(conv_layer->blobs()[1]->shape()));
				this->blobs_[1]->ShareData(*conv_layer->blobs()[1].get());
				this->blobs_[1]->ShareDiff(*conv_layer->blobs()[1].get());
			}
		}
	}

	template <typename Dtype>
	void ConvNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		conv_layer->Reshape(bottom, conv_top_vec);
		norm_bottom.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		caffe_set(norm_bottom.count(), (Dtype)1, norm_bottom.mutable_cpu_data());
		norm_layer->Reshape(norm_bottom_vec, norm_top_vec);
		norm_layer->Forward(norm_bottom_vec, norm_top_vec);
		top[0]->ReshapeLike(*conv_top_vec[0]);
	}

	template <typename Dtype>
	void ConvNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		conv_layer->Forward(bottom, conv_top_vec);
		for (int n = 0; n < conv_top_vec[0]->num(); n++)
		{
			caffe_div(norm_top.count(), conv_top_vec[0]->cpu_data() + conv_top_vec[0]->offset(n),
				norm_top.cpu_data(), top[0]->mutable_cpu_data() + top[0]->offset(n));
		}
	}

	template <typename Dtype>
	void ConvNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		caffe_set(conv_top_vec[0]->count(), (Dtype)0, conv_top_vec[0]->mutable_cpu_diff());
		for (int n = 0; n < conv_top_vec[0]->num(); n++)
		{
			caffe_div(norm_top.count(), top[0]->cpu_diff() + top[0]->offset(n),
				norm_top.cpu_data(), conv_top_vec[0]->mutable_cpu_diff()+conv_top_vec[0]->offset(n));
		}

		conv_layer->Backward(conv_top_vec, propagate_down, bottom);
	}

#ifdef CPU_ONLY
	STUB_GPU(ConvNormLayer);
#endif

	INSTANTIATE_CLASS(ConvNormLayer);
	REGISTER_LAYER_CLASS(ConvNorm);
}