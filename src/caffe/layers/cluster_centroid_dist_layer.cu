#include "caffe/layers/cluster_centroid_dist_layer.hpp"
#include <vector>

namespace caffe{
	//template <typename Dtype>
	//__global__ void eucliean_vec_forward(const int n, const int num, const int num_cluster,
	//	const int feat_dim, const Dtype* bottom_data, const Dtype* centroid_data,
	//	Dtype* top_data)
	//{
	//	CUDA_KERNEL_LOOP(index, n) {
	//		int k_idx = n % num_cluster;
	//		int n_idx = n / num_cluster;

	//		const Dtype* bottom_ptr = bottom_data + n_idx*feat_dim;
	//		const Dtype* centroid_ptr = centroid_data + k_idx*feat_dim;
	//		Dtype* top_ptr = top_data + n;
	//		top_ptr[0] = 0;
	//		for (int i = 0; i < feat_dim; ++i)
	//		{
	//			top_ptr[0] += 0.5*(bottom_ptr[i] - centroid_ptr[i])*(bottom_ptr[i] - centroid_ptr[i]);
	//		}

	//	}
	//}
	
	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* centroid_data = this->blobs_[0]->gpu_data();
		// square of data.
		caffe_gpu_mul(bottom[0]->count(), bottom_data, bottom_data, square_feat_.mutable_gpu_data());
		// sum along feat dim
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), 1, centroid_dim_,
			(Dtype)1.0, square_feat_.gpu_data(), ones_.gpu_data(), (Dtype)0.0, column_.mutable_gpu_data());
		//span along centroid num.
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), num_cluster_, 1,
			(Dtype)0.5, column_.gpu_data(), ones_.gpu_data(), (Dtype)0.0, cache_feat_.mutable_gpu_data());

		//square of centroid.
		caffe_gpu_mul(this->blobs_[0]->count(), centroid_data, centroid_data, square_cluster_.mutable_gpu_data());
		//sum along centroid dim
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_cluster_, 1, centroid_dim_,
			(Dtype)1.0, square_cluster_.gpu_data(), ones_.gpu_data(), (Dtype)0.0, column_.mutable_gpu_data());
		//span along feat num
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), num_cluster_, 1,
			(Dtype)0.5, ones_.gpu_data(), column_.gpu_data(), (Dtype)0.0, cache_cluster_.mutable_gpu_data());

		// dot product of centroid and feat
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, bottom[0]->num(), num_cluster_, centroid_dim_,
			(Dtype)-1.0, bottom_data, centroid_data, (Dtype)0.0, top_data);

		//sum all distance.
		caffe_gpu_add(top[0]->count(), cache_feat_.gpu_data(), top_data, top_data);
		caffe_gpu_add(top[0]->count(), cache_cluster_.gpu_data(), top_data, top_data);
		caffe_gpu_scale(top[0]->count(), (Dtype)scale, top_data, top_data);
		channel_div_kernel<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			top[0]->count(), num_cluster_, 1, top_data, this->blobs_[1]->gpu_data(), top_data);
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

		//caffe_gpu_sub(diff_.count(), diff_.gpu_data(), tmp_.gpu_data(), diff_.mutable_gpu_data());
		//caffe_gpu_mul(diff_.count(), diff_.gpu_data(), diff_.gpu_data(), tmp_.mutable_gpu_data());
		//caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, top[0]->count(), 1, centroid_dim_,
		//	(Dtype)1.0, tmp_.gpu_data(), ones_.gpu_data(), (Dtype)0.0, top_data);

		//for (int n = 0; n < bottom[0]->num(); n++)
		//{
		//	const Dtype* centroid_data = this->blobs_[0]->gpu_data();
		//	for (int k = 0; k < num_cluster_; ++k)
		//	{
		//		caffe_gpu_sub(centroid_dim_, bottom_data + bottom[0]->offset(n),
		//			centroid_data + k*centroid_dim_, diff_data);

		//		Dtype* top_ptr = top_data + (n*num_cluster_ + k);
		//		caffe_gpu_dot(centroid_dim_, diff_data, diff_data, top_ptr);
		//		*top_ptr = *top_ptr / (Dtype)centroid_dim_;
		//	}
		//}
	}


	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		channel_div_kernel<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
			top[0]->count(), 
			num_cluster_, 
			1, 
			top[0]->gpu_diff(), 
			this->blobs_[1]->gpu_data(), 
			temp_diff_.mutable_gpu_data());

		Dtype* centroid_diff = this->blobs_[0]->mutable_gpu_diff();
		const Dtype* top_diff = temp_diff_.gpu_data();
		const Dtype* centroid_data = this->blobs_[0]->gpu_data();
		const Dtype* top_data = top[0]->gpu_data();
		//sum diff along centroid num
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, top[0]->num(), 1, num_cluster_,
			(Dtype)1.0*scale, top_diff, ones_.gpu_data(), (Dtype)0.0, column_.mutable_gpu_data());
		//span diff along feat dim
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, top[0]->num(), centroid_dim_, 1,
			(Dtype)1.0, column_.gpu_data(), ones_.gpu_data(), (Dtype)0.0, square_feat_.mutable_gpu_data());
		//multipy feat data
		caffe_gpu_mul(bottom[0]->count(), square_feat_.gpu_data(), bottom[0]->gpu_data(), square_feat_.mutable_gpu_data());

		//sum diff along feat num
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_cluster_,top[0]->num(),
			(Dtype)1.0*scale, ones_.gpu_data(), top_diff,(Dtype)0.0, column_.mutable_gpu_data());
		//span diff along centroid dim
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_cluster_, centroid_dim_, 1,
			(Dtype)1.0, column_.gpu_data(), ones_.gpu_data(), (Dtype)0.0, square_cluster_.mutable_gpu_data());
		//multipy centroid data
		caffe_gpu_mul(this->blobs_[0]->count(), square_cluster_.gpu_data(), this->blobs_[0]->gpu_data(), square_cluster_.mutable_gpu_data());

		//dot diff of feat
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, top[0]->num(), centroid_dim_, num_cluster_,
			(Dtype)-1.0*scale, top_diff, centroid_data, (Dtype)0.0, bottom_diff);
		caffe_gpu_add(bottom[0]->count(), bottom_diff, square_feat_.gpu_data(), bottom_diff);

		//dot diff of centroid
		caffe_gpu_gemm(CblasTrans, CblasNoTrans, num_cluster_, centroid_dim_, top[0]->num(),
			(Dtype)-1.0*scale, top_diff, top_data, (Dtype)0.0, centroid_diff);
		caffe_gpu_add(this->blobs_[0]->count(), centroid_diff, square_cluster_.gpu_data(), centroid_diff);

		//// extend n*k top_data into n*k*d temp_data
		//caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, top[0]->count(),centroid_dim_, 1,
		//	(Dtype)1.0, top_diff, ones_.gpu_data(), (Dtype)0.0, tmp_data);

		//// n*k*d top_diff multipy diff_data
		//caffe_gpu_mul(diff_.count(), tmp_data, diff_data, tmp_data);
		//
		//// sum along channel n to get gradient of centroid.
		//caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_cluster_*centroid_dim_, bottom[0]->num(),
		//	(Dtype)-1.0, ones_.gpu_data(), tmp_data, (Dtype)0.0, centroid_diff);

		////sum along channel k to get gradient of bottom.
		//for (int n = 0; n < bottom[0]->num(); n++)
		//{
		//	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, centroid_dim_, num_cluster_,
		//		(Dtype)1.0, ones_.gpu_data(), tmp_data + tmp_.offset(n), (Dtype)0.0, bottom_diff + bottom[0]->offset(n));
		//}

		//for (int n = 0; n < bottom[0]->num(); n++)
		//{
		//	const Dtype* centroid_data = this->blobs_[0]->gpu_data();
		//	Dtype* centroid_diff = this->blobs_[0]->mutable_gpu_diff();
		//	for (int k = 0; k < num_cluster_; ++k)
		//	{
		//		caffe_gpu_sub(centroid_dim_, bottom_data + bottom[0]->offset(n),
		//			centroid_data + k*centroid_dim_, diff_data);

		//		Dtype alpha = top_diff[n*num_cluster_ + k] / (Dtype)centroid_dim_;
		//		caffe_gpu_axpby(
		//			centroid_dim_,              // count
		//			alpha,                              // alpha
		//			diff_.cpu_data(),                   // a
		//			Dtype(1),                           // beta
		//			bottom_diff + bottom[0]->offset(n));  // b

		//		alpha = -1 * alpha;
		//		caffe_gpu_axpby(
		//			centroid_dim_,              // count
		//			alpha,                              // alpha
		//			diff_.cpu_data(),                   // a
		//			Dtype(1),                           // beta
		//			centroid_diff + k*centroid_dim_);  // b
		//	}
		//}
	}
	INSTANTIATE_LAYER_GPU_FUNCS(ClusterCentroidDistLayer);
}