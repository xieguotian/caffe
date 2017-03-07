#include <vector>

#include "caffe/layers/transpose_layer.hpp"

namespace caffe {
#define BLOCK_DIM 16

	// This kernel is optimized to ensure all global reads and writes are coalesced,
	// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
	// than the naive kernel below.  Note that the shared memory array is sized to 
	// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
	// so that bank conflicts do not occur when threads address the array column-wise.
	template <typename Dtype>
	__global__ void transpose(Dtype *odata, const Dtype *idata, int width, int height)
	{
		__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

		// read the matrix tile into shared memory
		unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
		unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
		if ((xIndex < width) && (yIndex < height))
		{
			unsigned int index_in = yIndex * width + xIndex;
			block[threadIdx.y][threadIdx.x] = idata[index_in];
		}

		__syncthreads();

		// write the transposed matrix tile to global memory
		xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
		yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
		if ((xIndex < height) && (yIndex < width))
		{
			unsigned int index_out = yIndex * height + xIndex;
			odata[index_out] = block[threadIdx.x][threadIdx.y];
		}
	}

	template <typename Dtype>
	void TransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		dim3 grid(ceil((Dtype)size_x_ / (Dtype)BLOCK_DIM), ceil((Dtype)size_y_ / (Dtype)BLOCK_DIM), 1);
		dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
		transpose << <grid, threads >> >(top[0]->mutable_gpu_data(), bottom[0]->gpu_data(), size_x_, size_y_);
	}
	template <typename Dtype>
	void TransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		dim3 grid(ceil((Dtype)size_y_ / (Dtype)BLOCK_DIM), ceil((Dtype)size_x_ / (Dtype)BLOCK_DIM), 1);
		dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
		transpose << <grid, threads >> >(bottom[0]->mutable_gpu_diff(), top[0]->gpu_diff(), size_y_, size_x_);
	}
	INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);
}  // namespace caffe
