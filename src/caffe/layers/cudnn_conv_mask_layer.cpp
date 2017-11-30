#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_conv_mask_layer.hpp"
#include "caffe/filler.hpp"
namespace caffe {
	template <> map<string, shared_ptr<Blob<double> > > CuDNNConvolutionMaskLayer<double>::thread_caches_ = map<string, shared_ptr<Blob<double> > >();
	template <> map<string, shared_ptr<Blob<float> > > CuDNNConvolutionMaskLayer<float>::thread_caches_ = map<string, shared_ptr<Blob<float> > >();

	template <> map<string, void *> CuDNNConvolutionMaskLayer<double>::workspaceData=map<string, void *>();
	template <> map<string, void *> CuDNNConvolutionMaskLayer<float>::workspaceData=map<string, void *>();

	template <> map<string, size_t>  CuDNNConvolutionMaskLayer<double>::workspaceSizeInBytes=map<string, size_t>();
	template <> map<string, size_t>  CuDNNConvolutionMaskLayer<float>::workspaceSizeInBytes=map<string, size_t>();
	boost::mutex caches_mutex_;

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
	  this->layer_param_.convolution_param().weight_filler()));
  vector<int> weight_shape = this->blobs_[0]->shape();
  weight_shape[0] = weight_shape[0] / 9;
  weight_shape[2] = weight_shape[2] * 9;
  Blob<Dtype> tmp_weight;
  tmp_weight.Reshape(weight_shape);
  weight_filler->Fill(&tmp_weight);
  caffe_copy(tmp_weight.count(), tmp_weight.gpu_data(), this->blobs_[0]->mutable_gpu_data());
  //weight_filler->Fill(this->blobs_[0].get());

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
  //workspaceSizeInBytes = 0;
  //workspaceData = NULL;
  caches_mutex_.lock();
  thread_id_ = boost::lexical_cast<std::string>(boost::this_thread::get_id());
  if (workspaceData.find(thread_id_) == workspaceData.end())
  {
	  workspaceSizeInBytes[thread_id_] = 0;
	  workspaceData[thread_id_] = NULL;
  }
  caches_mutex_.unlock();

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
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
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
}

template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Base_Reshape(bottom, top);
  thread_id_ = boost::lexical_cast<std::string>(boost::this_thread::get_id());

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

  caches_mutex_.lock();
  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes[thread_id_]) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes[thread_id_] = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData[thread_id_]);

    cudaError_t err = cudaMalloc(&(this->workspaceData[thread_id_]), workspaceSizeInBytes[thread_id_]);
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
	  workspaceData[thread_id_] = NULL;
	  workspaceSizeInBytes[thread_id_] = 0;
    }

  }
  caches_mutex_.unlock();
  //// if we succeed in the allocation, set pointer aliases for workspaces
  for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
	  workspace[g] = reinterpret_cast<char *>(workspaceData[thread_id_])+g*max_workspace;
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }

  // Reshape top and mask
  mask_caches_.resize(top.size());
  vector<int> top_shape = top[0]->shape();
  //int channel_new = top[0]->shape(this->channel_axis_) / 9;
  //top_shape[this->channel_axis_] = channel_new;
  for (int i = 0; i < top.size(); ++i) 
  {
	  //top_shape[this->channel_axis_ + 1] /= 9;
	  //top_shape[this->channel_axis_ + 2] /= 9;
	  //top[i]->Reshape(top_shape);
	  mask_caches_[i].reset(new Blob<unsigned char>());
	  mask_caches_[i]->Reshape(top_shape);
  }

  caches_mutex_.lock();
  if (thread_caches_.find(thread_id_) == thread_caches_.end())
  {
	  thread_caches_[thread_id_].reset(new Blob<Dtype>()); //= shared_ptr<Blob<Dtype>>(new Blob<Dtype>());//.reset(new Blob<Dtype>());
  }
  caches_mutex_.unlock();

  shared_ptr<Blob<Dtype> > caches_;
  caches_ = thread_caches_[thread_id_];
  for (int i = 0; i < top.size(); ++i)
  {
	  if (caches_->count() < 9 * top[i]->count())
	  {
		  vector<int> shape = top[i]->shape();
		  shape[1] = shape[1] * 9;
		  caches_->Reshape(shape);
	  }
  }
}

template <typename Dtype>
CuDNNConvolutionMaskLayer<Dtype>::~CuDNNConvolutionMaskLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }
  thread_id_ = boost::lexical_cast<std::string>(boost::this_thread::get_id());
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
  caches_mutex_.lock();
  if (workspaceData[thread_id_] != NULL)
  {
	  cudaFree(workspaceData[thread_id_]);
	  workspaceData[thread_id_] = NULL;
  }
  thread_caches_[thread_id_].reset();
  caches_mutex_.unlock();
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::Base_Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	const int first_spatial_axis = this->channel_axis_ + 1;
	CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + this->num_spatial_axes_)
		<< "bottom num_axes may not change.";
	this->num_ = bottom[0]->count(0, this->channel_axis_);
	CHECK_EQ(bottom[0]->shape(this->channel_axis_), this->channels_)
		<< "Input size incompatible with convolution kernel.";
	// TODO: generalize to handle inputs of different shapes.
	for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
		CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
			<< "All inputs must have the same shape.";
	}
	// Shape the tops.
	this->bottom_shape_ = &bottom[0]->shape();
	this->compute_output_shape();
	vector<int> top_shape(bottom[0]->shape().begin(),
		bottom[0]->shape().begin() + this->channel_axis_);
	top_shape.push_back(this->num_output_);
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		top_shape.push_back(this->output_shape_[i]);
	}
	top_shape[this->channel_axis_] = top_shape[this->channel_axis_] / 9;
	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->Reshape(top_shape);
	}
	//if (reverse_dimensions()) {
	//	//conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
	//}
	//else {
	//	//conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
	//}
	//col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
	//output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
	// Setup input dimensions (conv_input_shape_).
	vector<int> bottom_dim_blob_shape(1, this->num_spatial_axes_ + 1);
	this->conv_input_shape_.Reshape(bottom_dim_blob_shape);
	int* conv_input_shape_data = this->conv_input_shape_.mutable_cpu_data();
	for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
		if (this->reverse_dimensions()) {
			//not inverse in convolution mask
			conv_input_shape_data[i] = top[0]->shape(this->channel_axis_ + i);
		}
		else {
			conv_input_shape_data[i] = bottom[0]->shape(this->channel_axis_ + i);
		}
	}
	// The im2col result buffer will only hold one image at a time to avoid
	// overly large memory usage. In the special case of 1x1 convolution
	// it goes lazily unused to save memory.
	//col_buffer_shape_.clear();
	//col_buffer_shape_.push_back(kernel_dim_ * group_);
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		if (this->reverse_dimensions()) {
			this->col_buffer_shape_.push_back(this->input_shape(i + 1));
		}
		else {
			this->col_buffer_shape_.push_back(this->output_shape_[i]);
		}
	}
	//col_buffer_.Reshape(col_buffer_shape_);
	this->bottom_dim_ = bottom[0]->count(this->channel_axis_);
	this->top_dim_ = top[0]->count(this->channel_axis_) * 9;
	//num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
	//num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
	// Set up the all ones "bias multiplier" for adding biases by BLAS
	this->out_spatial_dim_ = top[0]->count(first_spatial_axis);
	//if (bias_term_) {
	//	vector<int> bias_multiplier_shape(1, out_spatial_dim_);
	//	bias_multiplier_.Reshape(bias_multiplier_shape);
	//	caffe_set(bias_multiplier_.count(), Dtype(1),
	//		bias_multiplier_.mutable_cpu_data());
	//}
}
INSTANTIATE_CLASS(CuDNNConvolutionMaskLayer);

}   // namespace caffe
#endif
