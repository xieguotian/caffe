#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_torch_layer.hpp"

namespace caffe{
	const int WARP_SIZE = 32;

	// The maximum number of threads in a block
	const int MAX_BLOCK_SIZE = 512;

	// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
	static int getNumThreads(int nElem) {
		int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
		for (int i = 0; i != 5; ++i) {
			if (nElem <= threadSizes[i]) {
				return threadSizes[i];
			}
		}
		return MAX_BLOCK_SIZE;
	}

	// Returns the index of the most significant 1 bit in `val`.
	__device__ __forceinline__ int getMSB(int val) {
		return 31 - __clz(val);
	}

	template <typename Dtype>
	struct Float2 {
		Dtype v1, v2;
		__device__ Float2() {}
		__device__ Float2(Dtype v1, Dtype v2) : v1(v1), v2(v2) {}
		__device__ Float2(Dtype v) : v1(v), v2(v) {}
		__device__ Float2<Dtype>& operator+=(const Float2<Dtype>& a) {
			v1 += a.v1;
			v2 += a.v2;
			return *this;
		}
	};

	template <typename Dtype>
	struct SumOp {
		__device__ SumOp(const Dtype* t ,const int* dim) : tensor(t),dims(dim) {}
		__device__ __forceinline__ Dtype operator()(int n, int ch, int x) {
			return tensor[(n*dims[1] + ch)*dims[2] + x];
		}
		const Dtype* tensor;
		const int* dims;
	};

	template <typename Dtype>
	struct VarOp {
		__device__ VarOp(Dtype m, const Dtype* t,const int* dim) : mean(m), tensor(t),dims(dim) {}
		__device__ __forceinline__ Dtype operator()(int n, int ch, int x) {
			Dtype val = tensor[(n*dims[1] + ch)*dims[2] + x];
			return (val - mean) * (val - mean);
		}
		const Dtype mean;
		const Dtype* tensor;
		const int* dims;
	};

	template <typename Dtype>
	struct GradOp {
		__device__ GradOp(Dtype m, const Dtype* i, const Dtype* g, int* dim1, int* dim2)
		: mean(m), input(i), gradOutput(g),dims1(dim1),dims2(dim2) {}
		__device__ __forceinline__ Float2<Dtype> operator()(int n, int ch, int x) {
			Dtype g = gradOutput[(n*dims1[1] + ch)*dims1[2] + x];
			Dtype c = input[(n*dims2[1] + ch)*dims2[2] + x] - mean;
			return Float2<Dtype>(g, g * c);
		}
		const Dtype mean;
		const Dtype* input;
		const Dtype* gradOutput;
		const int* dims1;
		const int* dims2;
	};

	// Sum across all threads within a warp
	template <typename Dtype>
	static __device__ __forceinline__ Dtype warpSum(Dtype val) {
#if __CUDA_ARCH__ >= 300
		for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
			val += __shfl_xor(val, 1 << i, WARP_SIZE);
		}
#else
		__shared__ Dtype values[MAX_BLOCK_SIZE];
		values[threadIdx.x] = val;
		__threadfence_block();
		const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
		for (int i = 1; i < WARP_SIZE; i++) {
			val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
		}
#endif
		return val;
	}

	template <typename Dtype>
	static __device__ __forceinline__ Float2<Dtype> warpSum(Float2<Dtype> value) {
		value.v1 = warpSum(value.v1);
		value.v2 = warpSum(value.v2);
		return value;
	}

	// Sum across (batch, x/y/z) applying Op() pointwise
	template<typename Dtype, typename Op>
	__device__ Dtype reduce(Op op, int ch, int* dims) {
		Dtype sum = (Dtype)0;
		for (int batch = 0; batch < dims[0]; ++batch) {
			for (int x = threadIdx.x; x < dims[2]; x += blockDim.x) {
				sum += op(batch, ch, x);
			}
		}

		// sum over NumThreads within a warp
		sum = warpSum(sum);

		// 'transpose', and reduce within warp again
		__shared__ Dtype shared[32];
		__syncthreads();
		if (threadIdx.x % WARP_SIZE == 0) {
			shared[threadIdx.x / WARP_SIZE] = sum;
		}
		if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
			// zero out the other entries in shared
			shared[threadIdx.x] = (Dtype)0;
		}
		__syncthreads();
		if (threadIdx.x / WARP_SIZE == 0) {
			sum = warpSum(shared[threadIdx.x]);
			if (threadIdx.x == 0) {
				shared[0] = sum;
			}
		}
		__syncthreads();

		// Everyone picks it up, should be broadcast into the whole gradInput
		return shared[0];
	}

	template<typename Dtype>
	__global__ void BatchNormalizationUpdateOutputInference_kernel(
		const Dtype* input,
		Dtype* output,
		Dtype* runningMean,
		Dtype* runningVar,
		const Dtype* weight,
		const Dtype* bias,
		Dtype epsilon,
		int num, int channels, int spatial_dim) {
		int dims[3] = { num, channels, spatial_dim };
		int plane = blockIdx.x;
		int batch = blockIdx.y;

		Dtype invstd = 1.0f / sqrt(runningVar[plane] + epsilon);
		Dtype mean = runningMean[plane];
		Dtype gamma = weight!=NULL ? weight[plane] : 1.0f;
		Dtype beta = bias!=NULL ? bias[plane] : 0.0f;

		// Write normalized and update the output
		for (int x = threadIdx.x; x < dims[2]; x += blockDim.x) {
			int idx = (batch*dims[1] + plane)*dims[2] + x;
			Dtype inp = input[idx];
			output[idx] = gamma * (inp - mean) * invstd + beta;
		}
	}

	template<typename Dtype>
	__global__ void BatchNormalizationUpdateOutput_kernel(
		const Dtype* input,
		Dtype* output,
		const Dtype* weight,
		const Dtype* bias,
		const Dtype epsilon,
		const Dtype momentum,
		Dtype* runningMean,
		Dtype* runningVar,
		Dtype* saveMean,
		Dtype* saveStd,
		int num, int channels,int spatial_dim) {

		int dims[3] = { num, channels, spatial_dim };

		int plane = blockIdx.x;
		int N = dims[0] * dims[2];

		Dtype norm = 1.0f / N;

		// Compute the mean and variance across (batch, x/y/z)
		Dtype mean = reduce<Dtype>(SumOp<Dtype>(input,dims), plane,dims) * norm;
		__syncthreads();
		Dtype varN = reduce<Dtype>(VarOp<Dtype>(mean, input,dims), plane,dims);
		Dtype invStd = 0.0f;
		if (varN != 0.0f || epsilon != 0.0f) {
			invStd = 1 / sqrt(varN * norm + epsilon);
		}

		// Save the mean, variance, and moving averages
		if (threadIdx.x == 0) {
			// Momentum based writeback
			Dtype unbiasedVar = varN / (N - 1);
			saveMean[plane] = mean;
			saveStd[plane] = invStd;
			runningMean[plane] = (1 - momentum) * runningMean[plane] + momentum * mean;
			runningVar[plane] = (1 - momentum) * runningVar[plane] + momentum * unbiasedVar;
		}

		// Write normalized and update the output
		Dtype gamma = weight!=NULL ? weight[plane] : 1.0f;
		Dtype beta = bias!=NULL ? bias[plane] : 0.0f;

		for (int batch = 0; batch < dims[0]; ++batch) {
			for (int x = threadIdx.x; x < dims[2]; x += blockDim.x) {
				int idx = (batch*dims[1] + plane)*dims[2] + x;
				Dtype inp = input[idx];
				output[idx] = gamma * (inp - mean) * invStd + beta;
			}
		}
	}

	template<typename Dtype>
	void BatchNormTorchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* input = bottom[0]->gpu_data();
		Dtype* output = top[0]->mutable_gpu_data();
		const Dtype* weight = is_affine_? NULL: this->blobs_[3]->gpu_data();
		Dtype* bias = NULL;
		if (!is_affine_ && has_bias_term_)
			bias = this->blobs_[4]->mutable_gpu_data();
		Dtype* runningMean = this->blobs_[0]->mutable_gpu_data();
		Dtype* runningVar = this->blobs_[1]->mutable_gpu_data();

		if (use_global_stats_) {
			dim3 blocks(channels_, bottom[0]->shape(0));
			dim3 threads(getNumThreads(spatial_dim_));

			BatchNormalizationUpdateOutputInference_kernel << <blocks, threads>> >(
				input, output, runningMean, runningVar, weight, bias, eps_,
				bottom[0]->num(),channels_,spatial_dim_);
		}
		else {
			Dtype* saveMean = mean_.mutable_gpu_data();
			Dtype* saveStd = variance_.mutable_gpu_data();
			dim3 blocks(channels_);
			dim3 threads(getNumThreads(spatial_dim_));
			BatchNormalizationUpdateOutput_kernel << <blocks, threads>> >(
				input, output, weight, bias, eps_, moving_average_fraction_, runningMean, runningVar,
				saveMean, saveStd,
				bottom[0]->num(), channels_, spatial_dim_);
		}
		//THCudaCheck(cudaGetLastError());
	}

	template<typename Dtype>
	__global__ void BatchNormalizationBackward_kernel(
		const Dtype* input,
		const Dtype* gradOutput,
		Dtype* gradInput,
		Dtype* gradWeight,
		Dtype* gradBias,
		const Dtype* weight,
		const Dtype* runningMean,
		const Dtype* runningVar,
		const Dtype* saveMean,
		const Dtype* saveStd,
		bool train,
		Dtype scale,
		Dtype eps,
		int num, int channels, int spatial_dim
		) {
		int dims[3] = { num, channels, spatial_dim };

		int plane = blockIdx.x;
		int N = dims[0] * dims[2];

		Dtype mean, stdVal;
		if (train) {
			mean = saveMean[plane];
			stdVal = saveStd[plane];
		}
		else {
			mean = runningMean[plane];
			stdVal = 1 / sqrt(runningVar[plane] + eps);
		}

		Dtype weightVal = weight!=NULL ? weight[plane] : 1.0f;
		Dtype norm = 1.0f / N;

		// Compute two values across (batch, x/y/z) in one pass:
		// 1. Sum(gradOutput)
		// 2. DotProduct(input - mean, gradOutput)
		Float2<Dtype> res = reduce<Float2<Dtype>>(GradOp<Dtype>(mean, input, gradOutput, dims,dims), plane, dims);
		Dtype gradOutputSum = res.v1;
		Dtype dotP = res.v2;

		Dtype gradMean = gradOutputSum * norm;
		Dtype projScale = dotP * norm * stdVal * stdVal;
		Dtype gradScale = stdVal * weightVal;

		if (gradInput!=NULL) {
			for (int batch = 0; batch < dims[0]; ++batch) {
				for (int x = threadIdx.x; x <dims[2]; x += blockDim.x) {
					int idx = (batch*dims[1] + plane)*dims[2] + x;
					Dtype gradOut = gradOutput[idx];
					if (train) {
						Dtype inp = input[idx];
						Dtype proj = (inp - mean) * projScale;
						gradInput[idx] = (gradOut - proj - gradMean) * gradScale;
					}
					else {
						gradInput[idx] = gradOut * gradScale;
					}
				}
			}
		}

		if (gradWeight!=NULL) {
			if (threadIdx.x == 0) {
				gradWeight[plane] += scale * dotP * stdVal;
			}
		}

		if (gradBias!=NULL) {
			if (threadIdx.x == 0) {
				gradBias[plane] += scale * gradOutputSum;
			}
		}
	}

	template<typename Dtype>
	void BatchNormTorchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		const Dtype* input = bottom[0]->gpu_data();
		Dtype* gradInput = propagate_down[0] ? bottom[0]->mutable_gpu_diff() : NULL;
		const Dtype* gradOutput = top[0]->gpu_diff();
		Dtype* gradWeight = is_affine_ ? NULL : this->blobs_[3]->mutable_gpu_diff();
		Dtype* gradBias = NULL;
		if (!is_affine_ && has_bias_term_)
			gradBias = this->blobs_[4]->mutable_gpu_diff();
		const Dtype* weight = is_affine_ ? NULL : this->blobs_[3]->gpu_data();
		const Dtype* runningMean = this->blobs_[0]->gpu_data();
		const Dtype* runningVar = this->blobs_[1]->gpu_data();
		const Dtype* saveMean = mean_.gpu_data();
		const Dtype* saveStd = variance_.gpu_data();
		bool train = !use_global_stats_;


		dim3 blocks(channels_);
		dim3 threads(getNumThreads(spatial_dim_));
		Dtype scale = 1;
		BatchNormalizationBackward_kernel << <blocks, threads>> >(
			input, gradOutput, gradInput, gradWeight, gradBias, weight, runningMean, runningVar,
			saveMean, saveStd, train,scale, eps_,
			bottom[0]->num(), channels_, spatial_dim_);
		//THCudaCheck(cudaGetLastError());
	}

	INSTANTIATE_LAYER_GPU_FUNCS(BatchNormTorchLayer);
}