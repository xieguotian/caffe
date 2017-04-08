#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_torch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormTorchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter bn_param = this->layer_param_.batch_norm_param();
  const ScaleParameter& scale_param = this->layer_param_.scale_param();
  moving_average_fraction_ = 1- bn_param.moving_average_fraction();
  is_affine_ = bn_param.affine();
  use_global_stats_ = this->phase_ == TEST;
  if (bn_param.has_use_global_stats())
	  use_global_stats_ = bn_param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = bn_param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
	  vector<int> sz;
	  sz.push_back(channels_);
	  has_bias_term_ = false;
	  if (!is_affine_ && this->layer_param_.scale_param().bias_term())
	  {
		  this->blobs_.resize(5);

		  this->blobs_[3].reset(new Blob<Dtype>(sz));
		  //intial scale param
		  FillerParameter filler_param(scale_param.filler());
		  if (!scale_param.has_filler()) {
			  // Default to unit (1) filler for identity operation.
			  filler_param.set_type("constant");
			  filler_param.set_value(1);
		  }
		  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
		  filler->Fill(this->blobs_[3].get());

		  this->blobs_[4].reset(new Blob<Dtype>(sz));
		  //initial bias param.
		  shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(scale_param.bias_filler()));
		  bias_filler->Fill(this->blobs_[4].get());
		  has_bias_term_ = true;
	  }
	  else
	  {
		  if (!is_affine_)
		  {
			  this->blobs_.resize(4);
			  this->blobs_[3].reset(new Blob<Dtype>(sz));
			  //intial scale param
			  FillerParameter filler_param(scale_param.filler());
			  if (!scale_param.has_filler()) {
				  // Default to unit (1) filler for identity operation.
				  filler_param.set_type("constant");
				  filler_param.set_value(1);
			  }
			  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
			  filler->Fill(this->blobs_[3].get());
		  }
		  else
		  {
			  this->blobs_.resize(3);
		  }
	  }

    this->blobs_[0].reset(new Blob<Dtype>(sz));
	caffe_set(this->blobs_[0]->count(), Dtype(0),
	            this->blobs_[0]->mutable_cpu_data());
    this->blobs_[1].reset(new Blob<Dtype>(sz));
	caffe_set(this->blobs_[1]->count(), Dtype(1),
		this->blobs_[1]->mutable_cpu_data());
    sz[0]=1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
	caffe_set(this->blobs_[2]->count(), Dtype(0),
		this->blobs_[2]->mutable_cpu_data());
    //for (int i = 0; i < 3; ++i) {
    //  caffe_set(this->blobs_[i]->count(), Dtype(0),
    //            this->blobs_[i]->mutable_cpu_data());
    //}

  }
}

template <typename Dtype>
void BatchNormTorchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);

  sz[0]=bottom[0]->shape(0);

  spatial_dim_ = bottom[0]->count()/(channels_*bottom[0]->shape(0));
}

template <typename Dtype>
void BatchNormTorchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* input = bottom[0]->cpu_data();
	Dtype* output = top[0]->mutable_cpu_data();
	const Dtype* weight = this->blobs_[3]->cpu_data();
	Dtype* bias = NULL;
	if (has_bias_term_)
		bias = this->blobs_[4]->mutable_cpu_data();
	Dtype* runningMean = this->blobs_[0]->mutable_cpu_data();
	Dtype* runningVar = this->blobs_[1]->mutable_cpu_data();

	int nInput = bottom[0]->count() / spatial_dim_;
	if (use_global_stats_) {
#pragma omp parallel for
		for (int n = 0; n < nInput; n++)
		{
			const Dtype* input_data = input + n*spatial_dim_;
			Dtype* output_data = output + n*spatial_dim_;
			int ch = n % channels_;
			Dtype mean = runningMean[ch];
			Dtype invstd = 1 / sqrt(runningVar[ch] + eps_);
			Dtype w = weight == NULL ? 1 : weight[ch];
			Dtype b = bias == NULL ? 0 : bias[ch];

			for (int i = 0; i < spatial_dim_; i++)
			{
				output_data[i] = (input_data[i] - mean) * invstd * w + b;
			}
		}
	}
	else
	{
		NOT_IMPLEMENTED;
	}
}

template <typename Dtype>
void BatchNormTorchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormTorchLayer);
#endif

INSTANTIATE_CLASS(BatchNormTorchLayer);
REGISTER_LAYER_CLASS(BatchNormTorch);
}  // namespace caffe
