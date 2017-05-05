/**
* Author : yhyuan@pku.edu.cn
* Date :   2017/04/01
* Discription: implement the normed center loss function
*/

#include "caffe/layers/norm_center_loss_layer.hpp"


namespace caffe {
	template <typename Dtype>
	void NormCenterLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*> & bottom, const vector<Blob<Dtype>*> & top){
		Forward_cpu(bottom, top);
	}

	template<typename Dtype>
	void NormCenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		Backward_cpu(top, propagate_down, bottom);
	}
	INSTANTIATE_LAYER_GPU_FUNCS(NormCenterLossLayer);
}//namespace caffe