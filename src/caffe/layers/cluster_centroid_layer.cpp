#include "caffe/layers/cluster_centroid_layer.hpp"
#include "caffe/filler.hpp"
#include <vector>

namespace caffe{
	template <typename Dtype>
	void ClusterCentroidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
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
	}

	template <typename Dtype>
	void ClusterCentroidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		vector<int> top_shape(2);
		top_shape[0] = bottom[0]->num();
		top_shape[1] = centroid_dim_;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void ClusterCentroidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* centroid_data = this->blobs_[0]->cpu_data();
		for (int n = 0; n < bottom[0]->num(); n++)
		{
			int label = bottom_data[n];
			caffe_copy(centroid_dim_, centroid_data + label*centroid_dim_, 
				top_data+n*centroid_dim_);
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(ClusterCentroidLayer);
#endif

	INSTANTIATE_CLASS(ClusterCentroidLayer);
	REGISTER_LAYER_CLASS(ClusterCentroid);
}