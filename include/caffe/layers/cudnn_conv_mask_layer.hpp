#ifndef CAFFE_CUDNN_CONV_MASK_LAYER_HPP_
#define CAFFE_CUDNN_CONV_MASK_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"
#include <thread>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
namespace caffe {

#ifdef USE_CUDNN
	extern boost::mutex caches_mutex_;
/*
 * @brief cuDNN implementation of ConvolutionLayer.
 *        Fallback to ConvolutionLayer for CPU mode.
 *
 * cuDNN accelerates convolution through forward kernels for filtering and bias
 * plus backward kernels for the gradient w.r.t. the filters, biases, and
 * inputs. Caffe + cuDNN further speeds up the computation through forward
 * parallelism across groups and backward parallelism across gradients.
 *
 * The CUDNN engine does not have memory overhead for matrix buffers. For many
 * input and filter regimes the CUDNN engine is faster than the CAFFE engine,
 * but for fully-convolutional models and large inputs the CAFFE engine can be
 * faster as long as it fits in memory.
*/
template <typename Dtype>
class CuDNNConvolutionMaskLayer : public ConvolutionLayer<Dtype> {
 public:
	 explicit CuDNNConvolutionMaskLayer(const LayerParameter& param)
		 : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNConvolutionMaskLayer();

  virtual void Release_caches(){
	  for (int i = 0; i < mask_caches_.size(); ++i)
	  {
		  mask_caches_[i]->Release_mem();
	  }
	  LOG(INFO) << "Release caches";
  }

  virtual vector<shared_ptr<Blob<char>>> get_mask_caches(){ return mask_caches_; }
 protected:
	 virtual void Base_Reshape(const vector<Blob<Dtype>*>& bottom,
		 const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool handles_setup_;
  cudnnHandle_t* handle_;
  cudaStream_t*  stream_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t *fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;

  vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  vector<cudnnConvolutionDescriptor_t> conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  size_t *workspace_fwd_sizes_;
  size_t *workspace_bwd_data_sizes_;
  size_t *workspace_bwd_filter_sizes_;
  static map<string,size_t> workspaceSizeInBytes;  // size of underlying storage
  static map<string,void *> workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData

  static map <string, shared_ptr<Blob<Dtype>>> thread_caches_;
  string thread_id_;
  vector<shared_ptr<Blob<char>>> mask_caches_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
