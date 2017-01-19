#include "caffe/layers/cluster_centroid_layer.hpp"
#include <vector>

namespace caffe{
	template <typename Dtype>
	void ClusterCentroidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* centroid_data = this->blobs_[0]->gpu_data();
		for (int n = 0; n < bottom[0]->num(); n++)
		{
			int label = bottom_data[n];
			caffe_copy(centroid_dim_, centroid_data + label*centroid_dim_,
				top_data + n*centroid_dim_);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ClusterCentroidLayer);
}