#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_tree_layer.hpp"
#include "caffe\filler.hpp"
namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionTreeLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

	//*****************************************************
	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	num_layer_ = conv_param.num_layer_of_tree();
	ch_per_super_node_ = conv_param.num_channels_per_supernode();
	norm_tree_weight_ = conv_param.norm_tree_weight();
	shuffle_ = conv_param.shuffle();
	intermediate_output_ = conv_param.intermediate_output();
	num_spatial_per_supernode_ = conv_param.num_spatial_per_supernode();

	//const int* kernel_shape_data = kernel_shape_.cpu_data();
	const int* kernel_shape_data = this->kernel_shape_.cpu_data();
	use_spatial_ = intermediate_output_ > 0 && kernel_shape_data[0] > 1;
	if (!use_spatial_)
	{
		intermediate_output_ = channels_;
	}
	int max_num = std::max(intermediate_output_, num_output_);
	if (use_spatial_)
	{
		max_num = std::max(max_num, channels_*kernel_shape_data[0] * kernel_shape_data[1]);
	}

	// Re-Initialize and fill the weights:
	// output channels x input channels per-group x kernel height x kernel width
	re_weights_.ReshapeLike(*this->blobs_[0]);
	//re_weights2_.ReshapeLike(*this->blobs_[0]);
	if (!use_spatial_)
	{
		vector<int> shape_cache(2);
		shape_cache[0] = intermediate_output_; //channels_;
		shape_cache[1] = intermediate_output_; //channels_;
		re_weights_cache_.Reshape(shape_cache);
		re_weights_cache2_.Reshape(shape_cache);
		shape_cache[0] = num_output_;
		re_weights_cache3_.Reshape(shape_cache);
	}
	else
	{
		vector<int> shape_cache(1);
		shape_cache[0] = intermediate_output_*max_num; //channels_;
		re_weights_cache_.Reshape(shape_cache);
		re_weights_cache2_.Reshape(shape_cache);
		re_weights_cache3_.Reshape(shape_cache);
	}
	if (use_spatial_)
	{
		vector<int> shape_cache(2);
		shape_cache[0] = intermediate_output_;
		shape_cache[1] = channels_*kernel_shape_data[0] * kernel_shape_data[1];
		Sp_W_.Reshape(shape_cache);
	}

	//CHECK_EQ(channels_, num_output_);
	//weight_shape[0] = std::ceil(std::log2(Dtype(channels_ / group_)));
	//weight_shape[1] = 2 * num_output_;
	if (num_layer_ <= 0)
	{
		//num_layer_ = std::ceil(std::log2(Dtype(channels_ / group_)) / std::log2(ch_per_super_node_));
		num_layer_ = std::ceil(std::log2(Dtype(intermediate_output_ / group_)) / std::log2(ch_per_super_node_));
	}
	LOG(INFO) << "num_layer of conv_tree :" << num_layer_ << " channels per supernode: " << ch_per_super_node_
		<< "intermediate output: " << intermediate_output_ << "kernel_size: " << kernel_shape_data[0] << "num_spatial_per_node: " << num_spatial_per_supernode_;
	//connects_per_layer_ = ch_per_super_node_ * channels_;
	connects_per_layer_ = ch_per_super_node_ * intermediate_output_;

	vector<int> weight_shape(1);
	weight_shape[0] = (num_layer_ - 1)*connects_per_layer_ + num_output_*ch_per_super_node_;

	//weight_shape[1] = connects_per_layer_;
	//if (shuffle_)
	//{
	// LOG(INFO) << "shuffle using random shuffle among layers.";
	// vector<int> shape_tmp(2);
	// shape_tmp[0] = num_layer_;
	// shape_tmp[1] = channels_;
	// //this->blobs_.push_back(new Blob<Dtype>(shape_tmp));
	// this->blobs_.resize(this->blobs_.size()+1);
	// idx_blob_ = this->blobs_.size() - 1;
	// this->blobs_[idx_blob_].reset(new Blob<Dtype>(shape_tmp));
	// vector<int> idx_shuffle(channels_);
	// for (int i = 0; i < channels_; i++)
	//  idx_shuffle[i] = i;

	// for (int i = 0; i < num_layer_; i++)
	// {
	//  std::random_shuffle(idx_shuffle.begin(), idx_shuffle.end());
	//  Dtype* idx_ptr = this->blobs_[idx_blob_]->mutable_cpu_data() + i*channels_;
	//  for (int j = 0; j < channels_; j++)
	//  {
	//	  idx_ptr[j] = (Dtype)idx_shuffle[j];
	//  }
	// }
	//}


	Wp_.resize(num_layer_);
	Wpi_.resize(num_layer_);
	vector<int> shape_wp(2);
	//shape_wp[1] = channels_;
	shape_wp[1] = intermediate_output_;
	for (int i = 0; i < num_layer_; i++)
	{
		Wp_[i].reset(new Blob<Dtype>());
		Wpi_[i].reset(new Blob<Dtype>());
		//shape_wp[0] = channels_;
		//shape_wp[1] = channels_;
		shape_wp[0] = intermediate_output_;
		shape_wp[1] = intermediate_output_;

		Wp_[i]->Reshape(shape_wp);
		shape_wp[0] = num_output_;
		Wpi_[i]->Reshape(shape_wp);
		//Wp_[i]->ReshapeLike(*this->blobs_[0]);
		//Wpi_[i]->ReshapeLike(*this->blobs_[0]);
	}
	shape_wp[0] = num_output_;
	Wp_[num_layer_ - 1]->Reshape(shape_wp);
	//weight_shape[0] = conv_out_channels_;
	//weight_shape[1] = conv_in_channels_ / group_;
	//for (int i = 0; i < num_spatial_axes_; ++i) {
	// weight_shape.push_back(kernel_shape_data[i]);
	//}

  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
	  this->layer_param_.convolution_param().weight_filler()));

  if (false)
  {
	  //version 2 for initialization
	  for (int num_l = 0; num_l < num_layer_; num_l++)
	  {
		  vector<int> tmp_shape(2);
		  tmp_shape[0] = (num_l == (num_layer_ - 1)) ? num_output_ : intermediate_output_;
		  tmp_shape[1] = ch_per_super_node_;
		  Blob<Dtype> tmp_blob(tmp_shape);
		  tmp_blob.set_cpu_data(this->blobs_[0]->mutable_cpu_data() + num_l*intermediate_output_*ch_per_super_node_);
		  weight_filler->Fill(&tmp_blob);
	  }
  }
  else
  {
	  //version 1 for initialization
	  weight_filler->Fill(this->blobs_[0].get());
	  //caffe_gpu_powx(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), (Dtype)1 / (Dtype)num_layer_, this->blobs_[0]->mutable_gpu_data());
	  //int fan_in = channels_;
	  int fan_in = channels_ * kernel_shape_data[0] * kernel_shape_data[1];
	  Dtype n = fan_in;  // default to fan_in
	  Dtype std = std::pow(sqrt(Dtype(2) / n), use_spatial_ ? 1.0 / num_layer_ : 1.0 / (num_layer_ + 1));
	  caffe_rng_gaussian<Dtype>(this->blobs_[0]->count(), Dtype(0), std,
		  this->blobs_[0]->mutable_cpu_data());
	  sigma_ = std;

	  if (use_spatial_)
	  {
		  this->blobs_.resize(this->blobs_.size() + 1);
		  spatial_idx_ = this->blobs_.size() - 1;
		  vector<int> tmp_shape(1);
		  tmp_shape[0] = num_spatial_per_supernode_ * kernel_shape_data[0] * kernel_shape_data[1] * intermediate_output_;
		  this->blobs_[spatial_idx_].reset(new Blob<Dtype>(tmp_shape));
		  caffe_rng_gaussian<Dtype>(this->blobs_[spatial_idx_]->count(), Dtype(0), std,
			  this->blobs_[spatial_idx_]->mutable_cpu_data());
	  }
  }
  //*********************************************

  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[bottom.size()];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[bottom.size()];
  workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
  workspace_bwd_data_sizes_ = new size_t[bottom.size()];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];

  for (size_t i = 0; i < bottom.size(); ++i) {
    // initialize all to default algorithms
    fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
    bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
    bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
    // default algorithms don't require workspace
    workspace_fwd_sizes_[i] = 0;
    workspace_bwd_data_sizes_[i] = 0;
    workspace_bwd_filter_sizes_[i] = 0;
  }

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Set the indexing parameters.
  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.

  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      kernel_h, kernel_w);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;

  is_incremental_ = this->layer_param_.incremental() && this->phase_ == TRAIN;
  if (is_incremental_)
  {
	  this->blobs_.push_back(shared_ptr<Blob<Dtype> >()) ;
	  vector<int> idx_shape;
	  //idx_shape.push_back(direct_num_);
	  int idx_param_idx = this->blobs_.size() - 1;
	  this->blobs_[idx_param_idx].reset(new Blob<Dtype>(this->blobs_[0]->shape()));
	  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
		  this->layer_param_.convolution_param().weight_filler()));
	  weight_filler->Fill(this->blobs_[idx_param_idx].get());
	  w_history_.ReshapeLike(*this->blobs_[0]);
	  is_history_init_ = false;
  }

  if (this->layer_param_.convolution_param().is_binarized_param())
  {
	  sign_weight_.ReshapeLike(*this->blobs_[0]);
	  vector<int> shape;
	  shape.push_back(this->blobs_[0]->channels()*this->blobs_[0]->height()*this->blobs_[0]->width());
	  sum_cache_.Reshape(shape);
	  caffe_gpu_set(sum_cache_.count(), (Dtype)1.0, sum_cache_.mutable_gpu_data());
	  shape[0] = this->blobs_[0]->num();
	  sum_result_.Reshape(shape);
	  this->blobs_[0]->set_can_be_save_as_bin(true);
  }
}

template <typename Dtype>
void CuDNNConvolutionTreeLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
	if (is_direct_connect_)
		cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
		this->num_,
		this->num_output_ / this->group_, height_out, width_out,
		(this->num_output_ + direct_num_) * this->out_spatial_dim_,
		this->out_spatial_dim_, width_out, 1);
	else
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionTreeLayer<Dtype>::~CuDNNConvolutionTreeLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  cudaFree(workspaceData);
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionTreeLayer);

}   // namespace caffe
#endif
