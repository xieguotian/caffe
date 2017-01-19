#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/prob_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_max2(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out,Dtype* position) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
	Dtype pos = -1;
    for (int c = 0; c < channels; ++c) {
		if (data[(n * channels + c) * spatial_dim + s]>maxval)
		{
			maxval = data[(n * channels + c) * spatial_dim + s];
			pos = c;
		}
    }
	position[index] = pos;
    out[index] = maxval;
  }
}
template <typename Dtype>
__global__ void kernel_channel_min(const int num, const int channels,
	const int spatial_dim, const Dtype* data, Dtype* out, Dtype* position) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		Dtype maxval = FLT_MAX;
		Dtype pos = -1;
		for (int c = 0; c < channels; ++c) {
			if (data[(n * channels + c) * spatial_dim + s]<maxval)
			{
				maxval = data[(n * channels + c) * spatial_dim + s];
				pos = c;
			}
		}
		position[index] = pos;
		out[index] = maxval;
	}
}
template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_add(const int count,
	const int num, const int channels,
	const int spatial_dim, const Dtype* channel_max, Dtype* data) {
	CUDA_KERNEL_LOOP(index, count) {
		int n = index / channels / spatial_dim;
		int s = index % spatial_dim;
		data[index] += channel_max[n * spatial_dim + s];
	}
}

template <typename Dtype>
__global__ void kernel_channel_sum2(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += abs(data[(n * channels + c) * spatial_dim + s]);
    }
	channel_sum[index] = abs(sum); 
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum_subtract(const int num, const int channels,
	const int spatial_dim, const Dtype* data, const Dtype* position, Dtype* channel_sum) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		Dtype sum = 0;
		for (int c = 0; c < channels; ++c) {
			sum += (data[(n * channels + c) * spatial_dim + s]);
		}

		int idx = (n*channels + position[index])* spatial_dim + s;
		channel_sum[idx] -= sum;
	}
}
template <typename Dtype>
__global__ void kernel_channel_sum_add(const int num, const int channels,
	const int spatial_dim, const Dtype* data, const Dtype* position, Dtype* channel_sum) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		Dtype sum = 0;
		for (int c = 0; c < channels; ++c) {
			sum += (data[(n * channels + c) * spatial_dim + s]);
		}

		int idx = (n*channels + position[index])* spatial_dim + s;
		channel_sum[idx] += sum;
	}
}
template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void ProbNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_data = bottom[0]->mutable_gpu_data();;
	//if (use_T_)
	//{
	//	bottom_data = bottom[0]->mutable_gpu_diff();
	//	caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), bottom_data);
	//	caffe_gpu_scal(bottom[0]->count(), (Dtype)1.0 / temperature_, bottom_data);
	//}
	//else
	//{
	//	bottom_data = bottom[0]->mutable_gpu_data();
	//}
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  Dtype* position_data = position_.mutable_gpu_data();

  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_min<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
	  scale_data, position_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  //kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //    count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum2<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);

  if (use_T_)
  {
	  caffe_gpu_scal(count, (Dtype)1.0 / temperature_, top_data);
	  //caffe_gpu_set(bottom[0]->count(), (Dtype)0.0, bottom[0]->mutable_gpu_diff());
  }
}

template <typename Dtype>
void ProbNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  Dtype* scale_diff = scale_.mutable_gpu_diff();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  //caffe_gpu_scal(count, (Dtype)-1, bottom_diff);
  if (use_T_)
  {
	  caffe_gpu_scal(count, (Dtype)1.0 / temperature_, bottom_diff);
  }
  //// Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  //// NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_,
	  top_diff, top_data, scale_diff);
  
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype> << <CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
	  scale_diff, bottom_diff);
 
  kernel_channel_sum_subtract<Dtype> << <CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
	  CAFFE_CUDA_NUM_THREADS >> >(outer_num_, channels, inner_num_, bottom_diff, position_.gpu_data(),
	  bottom_diff);
  // elementwise multiplication
  //caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  kernel_channel_div<Dtype> << <CAFFE_GET_BLOCKS(count),
	  CAFFE_CUDA_NUM_THREADS >> >(count, outer_num_, channels, inner_num_,
	  scale_data, bottom_diff); 
  
}

INSTANTIATE_LAYER_GPU_FUNCS(ProbNormLayer);


}  // namespace caffe
