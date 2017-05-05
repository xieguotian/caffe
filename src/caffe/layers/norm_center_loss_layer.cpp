#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/norm_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> loss_shape_0(0);
		top[0]->Reshape(loss_shape_0);
		vector<int> loss_shape_1(0);
		top[1]->Reshape(loss_shape_1);

		const int num_output = this->layer_param_.norm_center_loss_param().num_output();
		N_ = num_output;
		const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.norm_center_loss_param().axis());
		// Dimensions starting from "axis" are "flattened" into a single
		// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
		// and axis == 1, N inner products with dimension CHW are performed.
		K_ = bottom[0]->count(axis);
		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		}
		else {
			this->blobs_.resize(2);
			// Intialize the weight
			vector<int> center_shape(2);
			center_shape[0] = N_;
			center_shape[1] = K_;
			this->blobs_[0].reset(new Blob<Dtype>(center_shape));
			this->blobs_[1].reset(new Blob<Dtype>(center_shape));
			// fill the weights
			shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
				this->layer_param_.norm_center_loss_param().center_filler()));
			center_filler->Fill(this->blobs_[0].get());
			center_filler->Fill(this->blobs_[1].get());
		}  // parameter initialization
		this->param_propagate_down_.resize(this->blobs_.size(), true);
		center_diff_.Reshape(N_, N_, K_, 1);
		vector<int> square_shape(2, 0);
		square_shape[0] = N_;
		square_shape[1] = K_;
		squared_.Reshape(square_shape);
		//init the iter_ as zero.
		iter_ = this->layer_param_.norm_center_loss_param().iteration();
	}

	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(bottom[1]->channels(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		M_ = bottom[0]->num();
		// The top shape will be the bottom shape with the flattened axes dropped,
		// and replaced by a single axis with dimension num_output (N_).
		LossLayer<Dtype>::Reshape(bottom, top);
		distance_.ReshapeLike(*bottom[0]);
		variation_sum_.ReshapeLike(*this->blobs_[0]);
	}

	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::norm_weight_forward_cpu(Dtype* weight, Dtype* norm_weight, 
	  int n, int d) {
	    Dtype* squared_data = squared_.mutable_cpu_data();
	    caffe_sqr<Dtype>(n*d, weight, squared_data);
	    for (int i = 0; i<n; ++i) {
	      Dtype normsqr = caffe_cpu_asum<Dtype>(d, squared_data + i*d);
	      caffe_cpu_scale<Dtype>(d, pow(normsqr, -0.5), weight + i*d, norm_weight + i*d);
	    }
	}

	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::norm_weight_backward_cpu(Dtype* top_diff, const Dtype* top_data, 
	  Dtype* bottom_diff, const Dtype* bottom_data, int n, int d) {
	    for (int i = 0; i < n; ++i) {
	      Dtype a = caffe_cpu_dot(d, top_data + i*d, top_diff + i*d);
	      caffe_cpu_scale(d, a, top_data + i*d, bottom_diff + i*d);
	      caffe_sub(d, top_diff + i*d, bottom_diff + i*d, bottom_diff + i*d);
	      a = caffe_cpu_dot(d, bottom_data + i*d, bottom_data + i*d);
	      caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff + i*d, bottom_diff + i*d);
	    }
	}

	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//dynamic weight function
		iter_ += (Dtype)1.;
		Dtype base_ = this->layer_param_.norm_center_loss_param().base();
		Dtype gamma_ = this->layer_param_.norm_center_loss_param().gamma();
		Dtype power_ = this->layer_param_.norm_center_loss_param().power();
		Dtype lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
		Dtype loss_weight = 1 / (1 + lambda_);

		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();
		Dtype* center = this->blobs_[0]->mutable_cpu_data();
		Dtype* norm_center = this->blobs_[1]->mutable_cpu_data();
		norm_weight_forward_cpu(center, norm_center, N_, K_);

		Dtype* distance_data = distance_.mutable_cpu_data();
		// loss for the (sample, centers)
		for (int i = 0; i < M_; i++) {
			const int label_value = static_cast<int>(label[i]);
			// D(i,:) = X(i,:) - C(y(i),:)
			caffe_sub(K_, bottom_data + i * K_, norm_center + label_value * K_, distance_data + i * K_);
		}
		Dtype dot = caffe_cpu_dot(M_ * K_, distance_.cpu_data(), distance_.cpu_data());
		Dtype loss = dot / M_ / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss * loss_weight;
		top[1]->mutable_cpu_data()[0] = loss_weight;
	}

	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype base_ = this->layer_param_.norm_center_loss_param().base();
		Dtype gamma_ = this->layer_param_.norm_center_loss_param().gamma();
		Dtype power_ = this->layer_param_.norm_center_loss_param().power();
		Dtype lambda_ = base_ * pow(((Dtype)1. + gamma_ * iter_), -power_);
		Dtype loss_weight = 1 / (1 + lambda_);

		// Gradient with respect to center
		const Dtype* center = this->blobs_[0]->cpu_data();
		Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
		const Dtype* norm_center = this->blobs_[1]->cpu_data();
		Dtype* norm_center_diff = this->blobs_[1]->mutable_cpu_diff();
		
		if (this->param_propagate_down_[0]) {
			const Dtype* label = bottom[1]->cpu_data();
			Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
			const Dtype* distance_data = distance_.cpu_data();
			// \sum_{y_i==j}
			caffe_set(N_ * K_, (Dtype)0., variation_sum_.mutable_cpu_data());
			for (int n = 0; n < N_; n++) {
				int count = 0;
				for (int m = 0; m < M_; m++) {
					const int label_value = static_cast<int>(label[m]);
					if (label_value == n) {
						count++;
						caffe_sub(K_, variation_sum_data + n * K_, distance_data + m * K_, variation_sum_data + n * K_);
					}
				}
				caffe_axpy(K_, (Dtype)1. / (count + (Dtype)1.), variation_sum_data + n * K_, norm_center_diff + n * K_);
				//dynamic weight loss
				caffe_scal(K_, loss_weight, norm_center_diff + n * K_);
			}
		}
		// Gradient with respect to bottom data 
		if (propagate_down[0]) {
			caffe_copy(M_ * K_, distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
			caffe_scal(M_ * K_, loss_weight * top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_cpu_diff());
		}
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		norm_weight_backward_cpu(norm_center_diff, norm_center, center_diff, center, N_, K_);
	}

#ifdef CPU_ONLY
	STUB_GPU(NormCenterLossLayer);
#endif
	INSTANTIATE_CLASS(NormCenterLossLayer);
	REGISTER_LAYER_CLASS(NormCenterLoss);
}  // namespace caffe