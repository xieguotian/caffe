#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cudnn_highway_layer.hpp"

namespace caffe {
	//=============== math function ===================================
	// Computes Y = H.G + X.(1 - G)

	template <typename Dtype>
	__global__ void kernel_gate_h_and_x_with_g(const int N, Dtype* H, const Dtype* X, Dtype* G, Dtype* Y) {
		CUDA_KERNEL_LOOP(index, N) {
			Y[index] = (H[index] * G[index]) + (X[index] * ((Dtype)1. - G[index]));
		}
	}

	template <typename Dtype>
	void caffe_gpu_gate_h_and_x_with_g(const int N, Dtype* H, const Dtype* X, Dtype* G, Dtype* Y) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_gate_h_and_x_with_g<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N, H, X, G, Y);
	}

	template void caffe_gpu_gate_h_and_x_with_g<float>(const int N,
		float* H, const float* X, float* G, float* Y);
	template void caffe_gpu_gate_h_and_x_with_g<double>(const int N,
		double* H, const double* X, double* G, double* Y);

	// Computes Y = A.(B - C)

	template <typename Dtype>
	__global__ void kernel_dot_with_diff(const int N, Dtype* A, const Dtype* B, const Dtype* C, Dtype* Y) {
		CUDA_KERNEL_LOOP(index, N) {
			Y[index] = A[index] * (B[index] - C[index]);
		}
	}

	template <typename Dtype>
	void caffe_gpu_dot_with_diff(const int N, Dtype* A, const Dtype* B,
		const Dtype* C, Dtype* Y) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_dot_with_diff<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N, A, B, C, Y);
	}

	template void caffe_gpu_dot_with_diff<float>(const int N, float* A, const float* B,
		const float* C, float* Y);
	template void caffe_gpu_dot_with_diff<double>(const int N, double* A, const double* B,
		const double* C, double* Y);

	// Computes Y = Y + A.(1 - B)

	template <typename Dtype>
	__global__ void kernel_dot_add_one_minus_b(const int N, const Dtype* A, const Dtype* B, Dtype* Y) {
		CUDA_KERNEL_LOOP(index, N) {
			Y[index] += A[index] * ((Dtype)1. - B[index]);
		}
	}

	template <typename Dtype>
	void caffe_gpu_dot_add_one_minus_b(const int N, const Dtype* A, const Dtype* B, Dtype* Y) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_dot_add_one_minus_b<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N, A, B, Y);
	}

	template void caffe_gpu_dot_add_one_minus_b<float>(const int N, const float* A, const float* B, float* Y);
	template void caffe_gpu_dot_add_one_minus_b<double>(const int N, const double* A, const double* B, double* Y);

	// elementwise multiply

	template <typename Dtype>
	__global__ void kernel_elem_multiply(const int N, const Dtype* A, const Dtype* B, Dtype* Y) {
		CUDA_KERNEL_LOOP(index, N) {
			Y[index] = A[index] * B[index];
		}
	}

	template <typename Dtype>
	void caffe_gpu_elem_multiply(const int N, const Dtype* A, const Dtype* B, Dtype* Y) {
		// NOLINT_NEXT_LINE(whitespace/operators)
		kernel_elem_multiply<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N, A, B, Y);
	}

	template void caffe_gpu_elem_multiply<float>(const int N, const float* A, const float* B, float* Y);
	template void caffe_gpu_elem_multiply<double>(const int N, const double* A, const double* B, double* Y);

	//======================================================
__global__ void sync_conv_groups_highway() { }

template <typename Dtype>
  __global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) {
      CUDA_KERNEL_LOOP(index, n) {
            out[index] = in[index] > Dtype(0) ? in[index] : Dtype(0);
      }
  }

template <typename Dtype>
__global__ void TanHForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}

template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    Dtype* cell_state_data = cell_states[i]->mutable_gpu_data();
    Dtype* transform_gate_data = transform_gate_states[i]->mutable_gpu_data();
    const Dtype* cell_weight = this->blobs_[0]->gpu_data();
    const Dtype* transform_weight = this->blobs_[1]->gpu_data();
    
    size_t workspace_limit_bytes = 2 * this->kernel_h_ *
                                   this->kernel_w_ *
                                   this->channels_ *
                                   sizeof(int) + 1;

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {

      // pick the convolution algorithm for the block state calculation
      // should be exposed in proto
      cudnnConvolutionFwdAlgo_t algo;

      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,  // memoryLimitInBytes,
        &algo));

      // pick the convolution algorithm for the transform gate calculation

      cudnnConvolutionFwdAlgo_t trans_algo;

      CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[g],
        bottom_descs_[i],
        trans_filter_desc_,
        trans_conv_descs_[i],
        top_descs_[i],
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes,  // memoryLimitInBytes,
        &trans_algo));

      // get minimum size of the workspace needed for the desired algorithm
      size_t workspaceSizeInBytes_temp = 0;
      size_t trans_workspaceSizeInBytes_temp = 0;

      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[g],
        bottom_descs_[i],
        filter_desc_,
        conv_descs_[i],
        top_descs_[i],
        algo,
        &workspaceSizeInBytes_temp));
      CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[g],
        bottom_descs_[i],
        trans_filter_desc_,
        trans_conv_descs_[i],
        top_descs_[i],
        trans_algo,
        &trans_workspaceSizeInBytes_temp));

      if ((workspaceSizeInBytes_temp + trans_workspaceSizeInBytes_temp)
          > workspaceSizeInBytes) {
        workspaceSizeInBytes = workspaceSizeInBytes_temp;
        // free the existing workspace and allocate a new (larger) one
        cudaFree(this->workspace);
        cudaError_t err = cudaMalloc(&(this->workspace), workspaceSizeInBytes);
        if (err != cudaSuccess) {
          // force zero memory path
          algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          trans_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
          workspace = NULL;
          workspaceSizeInBytes = 0;
        }
      }

      // Cells.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[0*this->group_ + g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], 
			bottom_data + bottom_offset_ * g,
            filter_desc_, 
			cell_weight + weight_offset_ * g,
            conv_descs_[i],
            algo, 
			workspace, 
			workspaceSizeInBytes,
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], 
			cell_state_data + top_offset_ * g));
      // Transform gates.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[1*this->group_ + g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], 
			bottom_data + bottom_offset_ * g,
            trans_filter_desc_, 
			transform_weight + trans_weight_offset_ * g,
            trans_conv_descs_[i],
            trans_algo, 
			workspace, 
			workspaceSizeInBytes,
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], 
			transform_gate_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* cell_bias_data = this->blobs_[2]->gpu_data();
        const Dtype* transform_bias_data = this->blobs_[3]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor_v2(handle_[0*this->group_ + g],
              CUDNN_ADD_SAME_C,
              cudnn::dataType<Dtype>::one,
              bias_desc_, cell_bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], cell_state_data + top_offset_ * g));
        CUDNN_CHECK(cudnnAddTensor_v2(handle_[1*this->group_ + g],
              CUDNN_ADD_SAME_C,
              cudnn::dataType<Dtype>::one,
              bias_desc_, transform_bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], transform_gate_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups_highway<<<1, 1>>>();

    const int count = bottom[i]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, cell_state_data, cell_state_data);
    CUDA_POST_KERNEL_CHECK;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, transform_gate_data, transform_gate_data);
    CUDA_POST_KERNEL_CHECK;

    // Finally, top = cell.input + bottom.(1 - input)
    caffe_gpu_gate_h_and_x_with_g(count, cell_state_data, bottom_data,
        transform_gate_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
        const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (in_data[index] > Dtype(0));
  }
}

template <typename Dtype>
__global__ void TanHBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}

template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* cell_weight = NULL;
  Dtype* cell_weight_diff = NULL;
  const Dtype* transform_weight = NULL;
  Dtype* transform_weight_diff = NULL;
  if (this->param_propagate_down_[0]) {

    cell_weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), cell_weight_diff);
    transform_weight_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), transform_weight_diff);
  }
  Dtype* cell_bias_diff = NULL;
  Dtype* transform_bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    cell_bias_diff = this->blobs_[2]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[2]->count(), Dtype(0), cell_bias_diff);
    transform_bias_diff = this->blobs_[3]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[3]->count(), Dtype(0), transform_bias_diff);
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* cell_state_data = cell_states[i]->gpu_data();
    const Dtype* transform_gate_data = transform_gate_states[i]->gpu_data();
    Dtype* cell_state_diff = cell_states[i]->mutable_gpu_diff();
    Dtype* transform_gate_diff = transform_gate_states[i]->mutable_gpu_diff();

    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    const Dtype* top_diff = top[i]->gpu_diff();
    const int count = top[i]->count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, cell_state_data, cell_state_diff);
    CUDA_POST_KERNEL_CHECK;
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, transform_gate_data, transform_gate_diff);
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_dot_with_diff(count, transform_gate_diff, cell_state_data,
        bottom_data, transform_gate_diff);
    caffe_gpu_elem_multiply(count, cell_state_diff, transform_gate_data,
        cell_state_diff);

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  
			  cell_state_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, 
			  cell_bias_diff + bias_offset_ * g));
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[3*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  
			  transform_gate_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, 
			  transform_bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter_v2(handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], 
			  bottom_data + bottom_offset_ * g,
              top_descs_[i],    
			  cell_state_diff + top_offset_ * g,
              conv_descs_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, 
			  cell_weight_diff + weight_offset_ * g));
        CUDNN_CHECK(cudnnConvolutionBackwardFilter_v2(handle_[4*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], 
			  bottom_data + bottom_offset_ * g,
              top_descs_[i],    
			  transform_gate_diff + top_offset_ * g,
              trans_conv_descs_[i],
              cudnn::dataType<Dtype>::one,
              trans_filter_desc_, 
			  transform_weight_diff + trans_weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (cell_weight == NULL || transform_weight == NULL) {
          cell_weight = this->blobs_[0]->gpu_data();
          transform_weight = this->blobs_[1]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData_v2(handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, 
			  cell_weight + weight_offset_ * g,
              top_descs_[i], 
			  cell_state_diff + top_offset_ * g,
              conv_descs_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], 
			  bottom_diff + bottom_offset_ * g));
        CUDNN_CHECK(cudnnConvolutionBackwardData_v2(handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              trans_filter_desc_, 
			  transform_weight + trans_weight_offset_ * g,
              top_descs_[i], 
			  transform_gate_diff + top_offset_ * g,
              trans_conv_descs_[i],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], 
			  bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups_highway<<<1, 1>>>();
    caffe_gpu_dot_add_one_minus_b(count, top_diff, transform_gate_data, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNHighwayLayer);

}  // namespace caffe
#endif
