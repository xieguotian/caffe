#include "caffe/layers/cluster_centroid_dist_layer.hpp"
#include "caffe/filler.hpp"
#include <vector>

namespace caffe{
	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
		scale = this->layer_param_.cluster_centroid_param().scale();
	}

	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		vector<int> top_shape(2);
		top_shape[0] = bottom[0]->num();
		top_shape[1] = num_cluster_;
		top[0]->Reshape(top_shape);

		cache_feat_.Reshape(top_shape);
		cache_cluster_.Reshape(top_shape);
		
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
	}

	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* centroid_data = this->blobs_[0]->cpu_data();

		// square of data.
		caffe_mul(bottom[0]->count(), bottom_data, bottom_data, square_feat_.mutable_cpu_data());
		// sum along feat dim
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), 1, centroid_dim_,
			(Dtype)1.0, square_feat_.cpu_data(), ones_.cpu_data(), (Dtype)0.0, column_.mutable_cpu_data());
		//span along centroid num.
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), num_cluster_, 1,
			(Dtype)0.5, column_.cpu_data(), ones_.cpu_data(), (Dtype)0.0, cache_feat_.mutable_cpu_data());

		//square of centroid.
		caffe_mul(this->blobs_[0]->count(), centroid_data, centroid_data, square_cluster_.mutable_cpu_data());
		//sum along centroid dim
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num_cluster_, 1, centroid_dim_,
			(Dtype)1.0, square_cluster_.cpu_data(), ones_.cpu_data(), (Dtype)0.0, column_.mutable_cpu_data());
		//span along feat num
		caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), num_cluster_, 1,
			(Dtype)0.5, ones_.cpu_data(), column_.cpu_data(), (Dtype)0.0, cache_cluster_.mutable_cpu_data());

		// dot product of centroid and feat
		caffe_cpu_gemm(CblasNoTrans, CblasTrans, bottom[0]->num(), num_cluster_, centroid_dim_,
			(Dtype)-1.0, bottom_data, centroid_data, (Dtype)0.0, top_data);

		//sum all distance.
		caffe_add(top[0]->count(), cache_feat_.cpu_data(), top_data, top_data);
		caffe_add(top[0]->count(), cache_cluster_.cpu_data(), top_data, top_data);
		caffe_cpu_scale(top[0]->count(), (Dtype)scale, top_data, top_data);
		//const Dtype* bottom_data = bottom[0]->cpu_data();
		//Dtype* top_data = top[0]->mutable_cpu_data();
		//Dtype* diff_data = diff_.mutable_cpu_data();

		//Dtype* tmp_data = tmp_.mutable_cpu_data();
		//const Dtype* centroid_data = this->blobs_[0]->cpu_data();
		//for (int n = 0; n < bottom[0]->num(); n++)
		//{
		//	for (int k = 0; k < num_cluster_; ++k)
		//	{
		//		caffe_copy(centroid_dim_, bottom_data + bottom[0]->offset(n), diff_data);
		//		diff_data += centroid_dim_;
		//	}
		//}
		//for (int n = 0; n < bottom[0]->num(); n++)
		//{
		//	caffe_copy(num_cluster_*centroid_dim_, centroid_data, tmp_data + tmp_.offset(n));
		//}

		//caffe_sub(diff_.count(), diff_.cpu_data(), tmp_.cpu_data(), diff_.mutable_cpu_data());
		//caffe_mul(diff_.count(), diff_.cpu_data(), diff_.cpu_data(), tmp_.mutable_cpu_data());
		//caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, top[0]->count(), 1, centroid_dim_,
		//	(Dtype)1.0, tmp_.cpu_data(), ones_.cpu_data(), (Dtype)0.0, top_data);

		//for (int n = 0; n < bottom[0]->num(); n++)
		//{
		//	const Dtype* centroid_data = this->blobs_[0]->cpu_data();
		//	for (int k = 0; k < num_cluster_; ++k)
		//	{
		//		caffe_sub(centroid_dim_, bottom_data + bottom[0]->offset(n),
		//			centroid_data + k*centroid_dim_, diff_data);
		//		top_data[n*num_cluster_ + k] = caffe_cpu_dot(centroid_dim_, diff_data, diff_data);
		//	}
		//}
	}

#ifdef CPU_ONLY
	STUB_GPU(ClusterCentroidDistLayer);
#endif

	INSTANTIATE_CLASS(ClusterCentroidDistLayer);
	REGISTER_LAYER_CLASS(ClusterCentroidDist);
}