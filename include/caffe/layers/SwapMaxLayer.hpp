#ifndef CAFFE_SWAPMAX_LAYER_HPP_
#define CAFFE_SWAPMAX__LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	template<typename Dtype>
	class SwapMaxLayer : public Layer<Dtype>{
	public:
		explicit SwapMaxLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top){}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top){
			top[0]->ReshapeLike(*bottom[0]);
		}

		virtual inline const char* type() const { return "SwapMax"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }

		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlos() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top){
			NOT_IMPLEMENTED;
		}


		/// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
			NOT_IMPLEMENTED;
		}
		
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
			NOT_IMPLEMENTED;
		}
	};


}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_