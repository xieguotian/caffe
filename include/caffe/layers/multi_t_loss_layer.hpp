#ifndef CAFFE_MULTI_T_LOSS_LAYER_HPP_
#define CAFFE_MULTI_T_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void caffe_gpu_dgmm(const CBLAS_SIDE mode, int M, int N, const Dtype *A,
		const Dtype *x, Dtype *C);

	template <typename Dtype>
	class MultiTLossLayer : public Layer<Dtype> {
	public:
		explicit MultiTLossLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 4; }
		virtual inline bool AutoTopBlobs() const { return true; }
		virtual inline const char* type() const { return "MultiTLoss"; }

		Blob<Dtype> *distance() { return &distance_; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		static const int TILE_DIM = 32;

		int N_, K_;
		bool sign_;

		Dtype lambda_;


		Blob<Dtype> distance_;
		Blob<Dtype> mask_;
		Blob<Dtype> coefm_;
		Blob<Dtype> coefn_;
		Blob<Dtype> count_;
		Blob<Dtype> diff_;
		Blob<Dtype> mean_;
		Blob<Dtype> sigma_prod_;
		Blob<Dtype> mu_sigma_;

	};
}
#endif