#ifndef CAFFE_PARAMETER_LAYER_HPP_
#define CAFFE_PARAMETER_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
namespace caffe {

template <typename Dtype>
class ParameterSelLayer : public Layer<Dtype> {
 public:
	 explicit ParameterSelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
      this->blobs_.resize(1);
      this->blobs_[0].reset(new Blob<Dtype>());
      this->blobs_[0]->Reshape(this->layer_param_.parameter_param().shape());

	  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
		  this->layer_param_.parameter_param().weight_filler()));
	  weight_filler->Fill(this->blobs_[0].get());
    }
    //top[0]->Reshape(this->layer_param_.parameter_param().shape());
  }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
	  vector<int> shape(this->layer_param_.parameter_param().shape().dim_size());
	  for (int i = 0; i < this->layer_param_.parameter_param().shape().dim_size(); ++i) {
		  shape[i] = this->layer_param_.parameter_param().shape().dim(i);
	  }
	  shape[0] = bottom[0]->num();
	  top[0]->Reshape(shape);
  }
  virtual inline const char* type() const { return "ParameterSel"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    /*top[0]->ShareData(*(this->blobs_[0]));
    top[0]->ShareDiff(*(this->blobs_[0]));*/
	  const Dtype* param_data;
	  Dtype* top_data;
	  if (Caffe::mode() == Caffe::GPU)
	  {
		  param_data = this->blobs_[0]->cpu_data();
		  top_data = top[0]->mutable_cpu_data();
	  }
	  else{
		  param_data = this->blobs_[0]->gpu_data();
		  top_data = top[0]->mutable_gpu_data();
	  }
	  int offset = this->blobs_[0]->offset(1);
	  for (int n = 0; n < bottom[0]->num(); n++)
	  {
		  if (bottom[0]->cpu_data()[n] >= 0 && bottom[0]->cpu_data()[n] < this->blobs_[0]->num())
		  {
			  int offset_tmp_param = bottom[0]->cpu_data()[n] * offset;
			  caffe_copy(offset, param_data + offset_tmp_param, top_data + n*offset);
		  }
		  else{
			  LOG(INFO) << "label: " << bottom[0]->cpu_data()[n];
		  }
	  }
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {
	  Dtype* param_data;
	  const Dtype* top_data;
	  if (Caffe::mode() == Caffe::GPU)
	  {
		  param_data = this->blobs_[0]->mutable_cpu_diff();
		  top_data = top[0]->cpu_diff();
	  }
	  else{
		  param_data = this->blobs_[0]->mutable_gpu_diff();
		  top_data = top[0]->gpu_diff();
	  }
	  int offset = this->blobs_[0]->offset(1);
	  for (int n = 0; n < bottom[0]->num(); n++)
	  {
		  if (bottom[0]->cpu_data()[n]>0 && bottom[0]->cpu_data()[n] < this->blobs_[0]->num())
		  {
			  int offset_tmp_param = bottom[0]->cpu_data()[n] * offset;
			  caffe_copy(offset, top_data + n*offset, param_data + offset_tmp_param);
		  }
	  }
  }
};

}  // namespace caffe

#endif
