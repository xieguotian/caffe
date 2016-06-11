#include <algorithm>
#include <vector>

#include "caffe/layers/binary_bounding_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"
namespace caffe {
	template<typename Dtype>
	void BinaryBoundingLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		this->blobs_.resize(1);
		vector<int> shape_tmp;
		shape_tmp.push_back(bottom[0]->channels());
		this->blobs_[0].reset(new Blob<Dtype>(shape_tmp));

		shared_ptr<Filler<Dtype>> centroid_filler(GetFiller<Dtype>(
			this->layer_param_.cluster_centroid_param().centroid_filler()));

		centroid_filler->Fill(this->blobs_[0].get());

		alpha = this->layer_param_.binary_bounding_param().alpha();
		beta = this->layer_param_.binary_bounding_param().beta();
		ratio = this->layer_param_.binary_bounding_param().ratio();
		dustbin_label = this->layer_param_.binary_bounding_param().dustbin_label();
		update_centroid = this->layer_param_.binary_bounding_param().update_centroid();
		threshold = this->layer_param_.binary_bounding_param().threshold();
		not_initialed = this->layer_param_.binary_bounding_param().not_initialed();
	}

	template<typename Dtype>
	void BinaryBoundingLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		LossLayer<Dtype>::Reshape(bottom, top);
		vector<int> shape;
		shape.push_back(bottom[0]->num());
		ones_column.Reshape(shape);
		cache_tmp_.ReshapeLike(*bottom[0]);
		square_cache_tmp_.ReshapeLike(*bottom[0]);
		scalar_cache_.ReshapeLike(*bottom[1]);
		ones_.ReshapeLike(*bottom[0]);

		caffe_set(ones_column.count(),
			(Dtype)1.0, 
			ones_column.mutable_cpu_data());
		caffe_set(ones_.count(),
			(Dtype)1.0, ones_.mutable_cpu_data());


	}
#ifdef CPU_ONLY
	STUB_GPU(SoftmaxLayer);
#endif

	INSTANTIATE_CLASS(BinaryBoundingLossLayer);
	REGISTER_LAYER_CLASS(BinaryBoundingLoss);
}