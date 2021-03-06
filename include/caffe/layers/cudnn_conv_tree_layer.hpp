#ifndef CAFFE_CUDNN_CONV_TREE_LAYER_HPP_
#define CAFFE_CUDNN_CONV_TREE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
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
class CuDNNConvolutionTreeLayer : public ConvolutionLayer<Dtype> {
 public:
  explicit CuDNNConvolutionTreeLayer(const LayerParameter& param)
      : ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~CuDNNConvolutionTreeLayer();

 protected:
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
  size_t workspaceSizeInBytes;  // size of underlying storage
  void *workspaceData;  // underlying storage
  void **workspace;  // aliases into workspaceData

  bool is_incremental_;
  Blob<Dtype> w_history_;
  bool is_history_init_;

  Blob<Dtype> sign_weight_;
  Blob<Dtype> sum_cache_;
  Blob<Dtype> sum_result_;
  Blob<Dtype> re_weights_;
  Blob<Dtype> re_weights_cache_;
  Blob<Dtype> re_weights_cache2_;
  Blob<Dtype> re_weights_cache3_;
  //Blob<Dtype> re_weights2_;
  int num_layer_;
  Dtype sigma_;

  vector<shared_ptr<Blob<Dtype>>> Wp_;
  vector<shared_ptr<Blob<Dtype>>> Wpi_;
  int ch_per_super_node_;
  int connects_per_layer_;
  bool norm_tree_weight_;
  bool shuffle_;
  int idx_blob_;
  Blob<Dtype> Sp_W_;
  int intermediate_output_;
  bool use_spatial_;
  int spatial_idx_;
  int num_spatial_per_supernode_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
