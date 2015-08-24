#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{
	template <typename Dtype>
	void DeconvNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
		
		LayerParameter deconv_param(this->layer_param());
		deconv_param.mutable_convolution_param()->set_bias_term(false);
		deconv_param.mutable_convolution_param()->mutable_weight_filler()->set_type("constant");
		deconv_param.mutable_convolution_param()->mutable_weight_filler()->set_value(0);
		deconv1_layer.reset(new DeconvolutionLayer<Dtype>(deconv_param));

		deconv1_bottom_vec.clear();
		deconv1_top_vec.clear();
		deconv1_bottom_vec.push_back(&constant1);
		//constant1.Reshape(1, 1, 1, 1);
		constant1.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		deconv1_top_vec.push_back(&coef_norm);
		deconv1_layer->SetUp(deconv1_bottom_vec, deconv1_top_vec);


		deconv_param = LayerParameter(this->layer_param());
		deconv_param.mutable_convolution_param()->set_bias_term(false);
		deconv2_layer.reset(new DeconvolutionLayer<Dtype>(deconv_param));
		deconv2_top_vec.clear();
		deconv2_top_vec.push_back(&deconv_val);
		deconv2_layer->SetUp(bottom, deconv2_top_vec);

		// Handle the parameters: weights and biases.
		// - blobs_[0] holds the filter weights
		// - blobs_[1] holds the biases (optional)
		bias_term_ = this->layer_param_.convolution_param().bias_term();

		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			if (bias_term_) {
				this->blobs_.resize(3);
			}
			else {
				this->blobs_.resize(2);
			}
			// Initialize and fill the weights:
			// output channels x input channels per-group x kernel height x kernel width
			this->blobs_[0].reset(new Blob<Dtype>(deconv2_layer->blobs()[0]->shape()));
			this->blobs_[1].reset(new Blob<Dtype>(deconv1_layer->blobs()[0]->shape()));
			this->blobs_[0]->CopyFrom(*deconv2_layer->blobs()[0]);
			//this->blobs_[1]->CopyFrom(*deconv1_layer->blobs()[0]);
			shared_ptr<Filler<Dtype>> alpha_filler(GetFiller<Dtype>(
				this->layer_param_.deconvolution_param().alpha_filler()));
			alpha_filler->Fill(this->blobs_[1].get());

			weights_alphas.reset(new Blob<Dtype>(deconv2_layer->blobs()[0]->shape()));
			alphas.reset(new Blob<Dtype>(deconv1_layer->blobs()[0]->shape()));

			Blob<Dtype> *alpha = alphas.get();
			Blob<Dtype> *wa = weights_alphas.get();

			deconv1_layer->blobs()[0]->ShareData(*alpha);
			deconv2_layer->blobs()[0]->ShareData(*wa);
			deconv1_layer->blobs()[0]->ShareDiff(*alpha);
			deconv2_layer->blobs()[0]->ShareDiff(*wa);
			// If necessary, initialize and fill the biases.
			if (bias_term_) {
				vector<int> bias_shape(1, this->layer_param_.convolution_param().num_output());
				this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
				shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
					this->layer_param_.convolution_param().bias_filler()));
				bias_filler->Fill(this->blobs_[2].get());
			}
		}
		LayerParameter exp_param;
		exp_top_vec.clear();
		exp_bottom_vec.clear();
		exp_bottom_vec.push_back(this->blobs_[1].get());
		exp_top_vec.push_back(alphas.get());
		exp_layer.reset(new ExpLayer<Dtype>(exp_param));
		exp_layer->SetUp(exp_bottom_vec, exp_top_vec);
		// Propagate gradients to the parameters (as directed by backward pass).
		this->param_propagate_down_.resize(this->blobs_.size(), true);
		average_train = true;
		if (this->layer_param_.has_deconvolution_param() && !this->layer_param_.deconvolution_param().average_train())
		{
			this->param_propagate_down_[1] = false;
			average_train = false;
		}
	}

	template <typename Dtype>
	void DeconvNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";

		constant1.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
		caffe_set(constant1.count(), Dtype(1), constant1.mutable_cpu_data());

		deconv2_layer->Reshape(bottom, deconv2_top_vec);
		deconv1_layer->Reshape(deconv1_bottom_vec, deconv1_top_vec);
		exp_layer->Reshape(exp_bottom_vec, exp_top_vec);

	    top[0]->ReshapeLike(*deconv2_top_vec[0]);

		deconv1_top_cache.Reshape(deconv1_top_vec[0]->shape());
		alpha_cache.Reshape(alphas->shape());
		alpha_cache2.Reshape(alphas->shape());
		// Set up the all ones "bias multiplier" for adding biases by BLAS
		if (bias_term_) {
			vector<int> bias_multiplier_shape(1, top[0]->height() * top[0]->width());
			bias_multiplier.Reshape(bias_multiplier_shape);
			caffe_set(bias_multiplier.count(), Dtype(1),
				bias_multiplier.mutable_cpu_data());
		}
	}

	template <typename Dtype>
	void DeconvNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		Dtype* wa = weights_alphas->mutable_cpu_data();

		exp_layer->Forward(exp_bottom_vec, exp_top_vec);
		for (int ch_in = 0; ch_in < weights_alphas->num(); ++ch_in)
		{
			caffe_mul(alphas->count(), this->blobs_[0]->cpu_data() + this->blobs_[0]->offset(ch_in),
				alphas->cpu_data(), wa + weights_alphas->offset(ch_in));
		}

		deconv2_layer->Forward(bottom, deconv2_top_vec);
		deconv1_layer->Forward(deconv1_bottom_vec, deconv1_top_vec);

		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* deconv1_top_vec_data = deconv1_top_vec[0]->cpu_data();
		const Dtype* deconv2_top_vec_data = deconv2_top_vec[0]->cpu_data();
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			caffe_div(deconv1_top_vec[0]->count(), deconv2_top_vec_data + deconv2_top_vec[0]->offset(n),
				deconv1_top_vec_data, top_data + top[0]->offset(n));

			if (this->bias_term_)
			{
				const Dtype* bias = this->blobs_[2]->cpu_data();
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, top[0]->channels(),
					top[0]->height() * top[0]->width(), 1, (Dtype)1., bias, bias_multiplier.cpu_data(),
					(Dtype)1., top_data + top[0]->offset(n));
			}
		}
	}

	template <typename Dtype>
	void DeconvNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* deconv1_top_vec_diff = deconv1_top_vec[0]->mutable_cpu_diff();
		Dtype* deconv2_top_vec_diff = deconv2_top_vec[0]->mutable_cpu_diff();
		const Dtype* deconv2_top_vec_data = deconv2_top_vec[0]->cpu_data();
		const Dtype* deconv1_top_vec_data = deconv1_top_vec[0]->cpu_data();

		caffe_set(deconv2_top_vec[0]->count(), (Dtype)0, deconv2_top_vec_diff);
		caffe_set(deconv1_top_vec[0]->count(), (Dtype)0, deconv1_top_vec_diff);
		caffe_set(exp_top_vec[0]->count(), (Dtype)0, exp_top_vec[0]->mutable_cpu_diff());
		//caffe_set(exp_bottom_vec[0]->count(), (Dtype)0, exp_bottom_vec[0]->mutable_cpu_diff());

		caffe_set(deconv1_layer->blobs()[0]->count(), (Dtype)0, deconv1_layer->blobs()[0]->mutable_cpu_diff());
		caffe_set(deconv2_layer->blobs()[0]->count(), (Dtype)0, deconv2_layer->blobs()[0]->mutable_cpu_diff());

		//bias gradient, if necessary
		if (this->bias_term_ && this->param_propagate_down_[2])
		{
			Dtype* bias_diff = this->blobs_[2]->mutable_cpu_diff();
			for (int n = 0; n < top[0]->num(); ++n)
			{
				caffe_cpu_gemv<Dtype>(CblasNoTrans, top[0]->channels(), top[0]->height() * top[0]->width(), 
					1., top_diff+top[0]->offset(n), bias_multiplier.cpu_data(), 1., bias_diff);
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
				caffe_div(deconv1_top_vec[0]->count(), top_diff + top[0]->offset(n),
					deconv1_top_vec_data, deconv2_top_vec_diff + deconv2_top_vec[0]->offset(n));
			}
			// backward throud deconv2_layer
			deconv2_layer->Backward(deconv2_top_vec, propagate_down, bottom);
			const Dtype* wa_diff = weights_alphas->cpu_diff();
			// weight gradient
			if (param_propagate_down_[0])
			{
				Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
				const Dtype* alpha = alphas->cpu_data();
				for (int ch_in = 0; ch_in < weights_alphas->num(); ++ch_in)
				{
					caffe_mul(alphas->count(), wa_diff + weights_alphas->offset(ch_in),
						alpha, weight_diff + this->blobs_[0]->offset(ch_in));
				}
			}

			// alpha gradient
			if (param_propagate_down_[1] && average_train)
			{
				//alpha_diff1
				Dtype* alpha_cache_diff = alpha_cache.mutable_cpu_diff();
				Dtype* alpha_cache_diff2 = alpha_cache2.mutable_cpu_diff();
				caffe_set(alpha_cache.count(), (Dtype)0, alpha_cache_diff);
				caffe_set(alpha_cache2.count(), (Dtype)0, alpha_cache_diff2);
				const Dtype* weight = this->blobs_[0]->cpu_data();
				for (int ch_in = 0; ch_in < weights_alphas->num(); ++ch_in)
				{
					caffe_mul(alpha_cache.count(), wa_diff + weights_alphas->offset(ch_in),
						weight + this->blobs_[0]->offset(ch_in), alpha_cache_diff);
					caffe_add(alpha_cache2.count(), alpha_cache_diff, alpha_cache_diff2, alpha_cache_diff2);
				}
				// top_diff backward to deonv1_top_vec_diff
				Dtype* deconv1_top_cache_diff = deconv1_top_cache.mutable_cpu_diff();
				caffe_set(deconv1_top_cache.count(), (Dtype)0, deconv1_top_cache_diff);
				for (int n = 0; n < top[0]->num(); ++n)
				{
					caffe_mul(deconv1_top_cache.count(), top_diff + top[0]->offset(n),
						deconv2_top_vec_data + deconv2_top_vec[0]->offset(n), deconv1_top_cache_diff);
					caffe_add(deconv1_top_cache.count(), deconv1_top_cache_diff, deconv1_top_vec_diff, deconv1_top_vec_diff);
				}
				caffe_div(deconv1_top_cache.count(), deconv1_top_vec_diff,
					deconv1_top_vec_data, deconv1_top_vec_diff);
				caffe_div(deconv1_top_cache.count(), deconv1_top_vec_diff,
					deconv1_top_vec_data, deconv1_top_vec_diff);
				// backward through deconv1_layer
				deconv1_layer->Backward(deconv1_top_vec, no_propagate_down, deconv1_bottom_vec);

				// alpha_diff2
				Dtype* alpha_diff = alphas->mutable_cpu_diff();
				//fuse alpha_diff1 and alpha_diff2
				caffe_sub(alpha_cache.count(), alpha_cache_diff2, alpha_diff, alpha_diff);

				exp_layer->Backward(exp_top_vec, yes_propagate_down, exp_bottom_vec);
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(DeconvNormLayer);
#endif

	INSTANTIATE_CLASS(DeconvNormLayer);
	REGISTER_LAYER_CLASS(DeconvNorm);
}