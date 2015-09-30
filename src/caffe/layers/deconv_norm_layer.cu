#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
	void DeconvNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Dtype* wa = weights_alphas->mutable_gpu_data();

		exp_layer->Forward(exp_bottom_vec, exp_top_vec);
		for (int ch_in = 0; ch_in < weights_alphas->num(); ++ch_in)
		{
			caffe_gpu_mul(alphas->count(), this->blobs_[0]->gpu_data() + this->blobs_[0]->offset(ch_in),
				alphas->gpu_data(), wa + weights_alphas->offset(ch_in));
		}
		deconv2_layer->Forward(bottom, deconv2_top_vec);
		deconv1_layer->Forward(deconv1_bottom_vec, deconv1_top_vec);

		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* deconv1_top_vec_data = deconv1_top_vec[0]->gpu_data();
		const Dtype* deconv2_top_vec_data = deconv2_top_vec[0]->gpu_data();
		caffe_gpu_add_scalar(deconv1_top_vec[0]->count(), (Dtype)std::numeric_limits<Dtype>::epsilon(),
			deconv1_top_vec[0]->mutable_gpu_data());
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			caffe_gpu_div(deconv1_top_vec[0]->count(), deconv2_top_vec_data + deconv2_top_vec[0]->offset(n),
				deconv1_top_vec_data, top_data + top[0]->offset(n));

			if (this->bias_term_)
			{
				const Dtype* bias = this->blobs_[2]->gpu_data();
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, top[0]->channels(),
					top[0]->height() * top[0]->width(), 1, (Dtype)1., bias, bias_multiplier.gpu_data(),
					(Dtype)1., top_data + top[0]->offset(n));
			}
		}
	}

	template <typename Dtype>
	void DeconvNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* deconv1_top_vec_diff = deconv1_top_vec[0]->mutable_gpu_diff();
		Dtype* deconv2_top_vec_diff = deconv2_top_vec[0]->mutable_gpu_diff();
		const Dtype* deconv2_top_vec_data = deconv2_top_vec[0]->gpu_data();
		const Dtype* deconv1_top_vec_data = deconv1_top_vec[0]->gpu_data();

		caffe_gpu_set(deconv2_top_vec[0]->count(), (Dtype)0, deconv2_top_vec_diff);
		caffe_gpu_set(deconv1_top_vec[0]->count(), (Dtype)0, deconv1_top_vec_diff);
		caffe_gpu_set(exp_top_vec[0]->count(), (Dtype)0, exp_top_vec[0]->mutable_gpu_diff());
		//caffe_gpu_set(exp_bottom_vec[0]->count(), (Dtype)0, exp_bottom_vec[0]->mutable_gpu_diff());

		caffe_gpu_set(deconv1_layer->blobs()[0]->count(), (Dtype)0, deconv1_layer->blobs()[0]->mutable_gpu_diff());
		caffe_gpu_set(deconv2_layer->blobs()[0]->count(), (Dtype)0, deconv2_layer->blobs()[0]->mutable_gpu_diff());

		//bias gradient, if necessary
		if (this->bias_term_ && this->param_propagate_down_[2])
		{
			Dtype* bias_diff = this->blobs_[2]->mutable_gpu_diff();
			for (int n = 0; n < top[0]->num(); ++n)
			{
				caffe_gpu_gemv<Dtype>(CblasNoTrans, top[0]->channels(), top[0]->height() * top[0]->width(),
					1., top_diff + top[0]->offset(n), bias_multiplier.gpu_data(), 1., bias_diff);
			}
		}

		// weights and alpha gradient, propagate down to bottom
		if (param_propagate_down_[0] || param_propagate_down_[1] || propagate_down[0])
		{
			vector<bool> no_propagate_down;
			no_propagate_down.push_back(false);
			vector<bool> yes_propagate_down;
			yes_propagate_down.push_back(true);
			// top_diff backward to deconv2_top_vec_diff
			for (int n = 0; n < top[0]->num(); ++n)
			{
				caffe_gpu_div(deconv1_top_vec[0]->count(), top_diff + top[0]->offset(n),
					deconv1_top_vec_data, deconv2_top_vec_diff + deconv2_top_vec[0]->offset(n));
			}
			// backward throud deconv2_layer
			deconv2_layer->Backward(deconv2_top_vec, propagate_down, bottom);
			const Dtype* wa_diff = weights_alphas->gpu_diff();
			// weight gradient
			if (param_propagate_down_[0])
			{
				Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
				const Dtype* alpha = alphas->gpu_data();
				for (int ch_in = 0; ch_in < weights_alphas->num(); ++ch_in)
				{
					caffe_gpu_mul(alphas->count(), wa_diff + weights_alphas->offset(ch_in),
						alpha, weight_diff + this->blobs_[0]->offset(ch_in));
				}
			}

			// alpha gradient
			if (param_propagate_down_[1] && average_train)
			{
				//alpha_diff1
				Dtype* alpha_cache_diff = alpha_cache.mutable_gpu_diff();
				Dtype* alpha_cache_diff2 = alpha_cache2.mutable_gpu_diff();
				caffe_gpu_set(alpha_cache.count(), (Dtype)0, alpha_cache_diff);
				caffe_gpu_set(alpha_cache2.count(), (Dtype)0, alpha_cache_diff2);
				const Dtype* weight = this->blobs_[0]->gpu_data();
				for (int ch_in = 0; ch_in < weights_alphas->num(); ++ch_in)
				{
					caffe_gpu_mul(alpha_cache.count(), wa_diff + weights_alphas->offset(ch_in),
						weight + this->blobs_[0]->offset(ch_in), alpha_cache_diff);
					caffe_gpu_add(alpha_cache2.count(), alpha_cache_diff, alpha_cache_diff2, alpha_cache_diff2);
				}
				// top_diff backward to deonv1_top_vec_diff
				Dtype* deconv1_top_cache_diff = deconv1_top_cache.mutable_gpu_diff();
				caffe_gpu_set(deconv1_top_cache.count(), (Dtype)0, deconv1_top_cache_diff);
				for (int n = 0; n < top[0]->num(); ++n)
				{
					caffe_gpu_mul(deconv1_top_cache.count(), top_diff + top[0]->offset(n),
						deconv2_top_vec_data + deconv2_top_vec[0]->offset(n), deconv1_top_cache_diff);
					caffe_gpu_add(deconv1_top_cache.count(), deconv1_top_cache_diff, deconv1_top_vec_diff, deconv1_top_vec_diff);
				}
				caffe_gpu_div(deconv1_top_cache.count(), deconv1_top_vec_diff,
					deconv1_top_vec_data, deconv1_top_vec_diff);
				caffe_gpu_div(deconv1_top_cache.count(), deconv1_top_vec_diff,
					deconv1_top_vec_data, deconv1_top_vec_diff);
				// backward through deconv1_layer
				deconv1_layer->Backward(deconv1_top_vec, no_propagate_down, deconv1_bottom_vec);

				// alpha_diff2
				Dtype* alpha_diff = alphas->mutable_gpu_diff();
				//fuse alpha_diff1 and alpha_diff2
				caffe_gpu_sub(alpha_cache.count(), alpha_cache_diff2, alpha_diff, alpha_diff);

				exp_layer->Backward(exp_top_vec, yes_propagate_down, exp_bottom_vec);
			}
		}
	}
	INSTANTIATE_LAYER_GPU_FUNCS(DeconvNormLayer);
}
