#include <vector>
#include "caffe/Filler.hpp"
#include "caffe/layers/scale_layer2.hpp"
#include "caffe/Blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template<typename Dtype>
	void Scale2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		num_output = this->layer_param_.convolution_param().num_output();

		vector<int> scale_shape(1, bottom[0]->channels());

		if (this->blobs_.size() > 0){
			if (scale_shape != this->blobs_[0]->shape()){
				Blob<Dtype> scale_shaped_blob(scale_shape);
				LOG(FATAL) << "Incorrect weight shape: expected shape "
					<< scale_shaped_blob.shape_string() << "; instead, shape was "
					<< this->blobs_[0]->shape_string();
			}

		}
		else{
			this->blobs_.resize(2);
		}

		this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
		shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(
			this->layer_param_.convolution_param().weight_filler()));
		scale_filler->Fill(this->blobs_[0].get());

		this->blobs_[1].reset(new Blob<Dtype>(scale_shape));
		caffe_set(this->blobs_[1]->count(), (Dtype)-1, this->blobs_[1]->mutable_cpu_data());
	}

	template<typename Dtype>
	void Scale2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		top[0]->Reshape(bottom[0]->num(), num_output,
			bottom[0]->height(), bottom[0]->width());
	}

	template<typename Dtype>
	void Scale2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		caffe_set<Dtype>(top[0]->count(), (Dtype)-1, top_data);

		const Dtype* scale_coeff = this->blobs_[0]->cpu_data();
		const Dtype* idx_coeff = this->blobs_[1]->cpu_data();

		//caffe_copy<Dtype>(bottom[0]->count(), bottom_data, top_data);

		int spat_dim = bottom[0]->count(2);

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				if (idx_coeff[ch] >= 0)
				{
					/*caffe_copy<Dtype>(spat_dim,
					bottom_data + (int)idx_coeff[ch] * spat_dim, top_data);

					caffe_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
					top_data);

					top_data += spat_dim;*/
					caffe_copy<Dtype>(spat_dim, bottom_data,
						top_data + (int)idx_coeff[ch] * spat_dim);
					caffe_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
						top_data + (int)idx_coeff[ch] * spat_dim);
				}
				bottom_data += spat_dim;

			}
			top_data += top[0]->count(1);
			//bottom_data += bottom[0]->count(1);
		}
	}
	template<typename Dtype>
	void Scale2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		const Dtype* scale_coeff = this->blobs_[0]->cpu_data();
		const Dtype* idx_coeff = this->blobs_[1]->cpu_data();

		int spat_dim = bottom[0]->count(2);
		//caffe_copy<Dtype>(bottom[0]->count(), top_diff, bottom_diff);

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			for (int ch = 0; ch < bottom[0]->channels(); ++ch)
			{
				if (idx_coeff[ch] >= 0)
				{
					/*caffe_copy<Dtype>(spat_dim, top_diff,
					bottom_diff + (int)idx_coeff[ch] * spat_dim);

					caffe_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
					bottom_diff + (int)idx_coeff[ch] * spat_dim);

					top_diff += spat_dim;*/
					caffe_copy<Dtype>(spat_dim,
						top_diff + (int)idx_coeff[ch] * spat_dim, bottom_diff);
					caffe_scal<Dtype>(spat_dim, (Dtype)scale_coeff[ch],
						bottom_diff);
				}
				bottom_diff += spat_dim;
			}
			top_diff += top[0]->count(1);
			//bottom_diff += bottom[0]->count(1);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(Scale2Layer);
#endif

	INSTANTIATE_CLASS(Scale2Layer);
	REGISTER_LAYER_CLASS(Scale2);
}