#include "caffe/layers/cluster_centroid_dist_layer.hpp"
#include "caffe/filler.hpp"
#include <vector>

namespace caffe{
	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		compute_dist_ = bottom.size() > 1;

		if (!compute_dist_)
		{
			centroid_dim_ = this->layer_param().cluster_centroid_param().dim();
			num_cluster_ = this->layer_param().cluster_centroid_param().num_cluster();

			vector<int> cluster_shape(2);
			cluster_shape[0] = num_cluster_;
			cluster_shape[1] = centroid_dim_;

			this->blobs_.resize(1);
			this->blobs_[0].reset(new Blob<Dtype>(cluster_shape));
			shared_ptr<Filler<Dtype>> cluster_filler(GetFiller<Dtype>(
				this->layer_param_.cluster_centroid_param().centroid_filler()));
			cluster_filler->Fill(this->blobs_[0].get());
			bool isinit = this->layer_param_.binary_bounding_param().not_initialed();
			initialized_ = false || (this->layer_param().phase() == TEST) || isinit;
			init_count_ = 0;

			use_T_ = this->layer_param_.softmax_param().temperature() != 0;
			T = this->layer_param_.softmax_param().temperature();

			is_sample_base_cls = this->layer_param_.binary_bounding_param().update_centroid();
		}

		scale = this->layer_param_.cluster_centroid_param().scale();
		this->param_propagate_down_.resize(this->blobs_.size(), true);

		is_self_dist_ = compute_dist_ && bottom[0]->data() == bottom[1]->data();
		use_square_ = this->layer_param_.binary_bounding_param().update_centroid();

	}

	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		if (compute_dist_)
		{
			centroid_dim_ = bottom[1]->channels();
			num_cluster_ = bottom[1]->num();
		}
		is_self_dist_ = compute_dist_ && bottom[0]->data() == bottom[1]->data();

		vector<int> top_shape(2);
		top_shape[0] = bottom[0]->num();
		//top_shape[1] = is_self_dist_ ? num_cluster_ - 1 : num_cluster_;
		top_shape[1] = num_cluster_;
		//if (is_self_dist_)
		//	LOG(INFO) << "shape:" << top_shape[0] << "," << top_shape[1];
		top[0]->Reshape(top_shape);
		if (!compute_dist_)
		{
			if (is_sample_base_cls)
			{
				vector<int> top_shape2(1);
				top_shape2[0] = bottom[0]->num();
				top[1]->Reshape(top_shape2);
			}
		}

		vector<int> square_shape(2);
		square_shape[0] = bottom[0]->num();
		square_shape[1] = bottom[0]->channels();
		square_feat_.Reshape(square_shape);
		square_shape[0] = num_cluster_;
		square_shape[1] = centroid_dim_;
		square_cluster_.Reshape(square_shape);

		vector<int> ones_shape(1);
		ones_shape[0] = std::max(bottom[0]->num(), std::max(num_cluster_, centroid_dim_));
		ones_.Reshape(ones_shape);
		caffe_set(ones_.count(), (Dtype)1.0, ones_.mutable_cpu_data());

		vector<int> column_shape(1);
		column_shape[0] = std::max(bottom[0]->num(), num_cluster_);
		column_.Reshape(column_shape);

		top_shape[0] = bottom[0]->num();
		top_shape[1] = num_cluster_;
		top_cache_.Reshape(top_shape);
		bottom_cache_.ReshapeLike(*bottom[0]);
		num_samp_ = bottom[0]->num();

	}

	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		NOT_IMPLEMENTED;
	}

#ifdef CPU_ONLY
	STUB_GPU(ClusterCentroidDistLayer);
#endif

	INSTANTIATE_CLASS(ClusterCentroidDistLayer);
	REGISTER_LAYER_CLASS(ClusterCentroidDist);
}