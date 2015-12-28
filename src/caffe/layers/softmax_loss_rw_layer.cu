#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/softmax_with_loss_rw_layer.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void SoftmaxLossForwardGPU2(const int nthreads,
		const Dtype* prob_data, const Dtype* label, Dtype* loss,
		const int num, const int dim, const int spatial_dim,
		Dtype* counts) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int n = index / spatial_dim;
			const int s = index % spatial_dim;
			const int label_value = static_cast<int>(label[n * spatial_dim + s]);

				int idx = n * dim + label_value * spatial_dim + s;
				loss[idx] = -log(max(prob_data[idx],
					Dtype(FLT_MIN)));
				counts[idx] = 1;
		}
	}

	template <typename Dtype>
	void SoftmaxWithLossRwLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
		const Dtype* prob_data = prob_.gpu_data();
		const Dtype* label = bottom[1]->gpu_data();
		const int dim = prob_.count() / outer_num_;
		const int nthreads = outer_num_ * inner_num_;
		// Since this memory is not used for anything until it is overwritten
		// on the backward pass, we use it here to avoid having to allocate new GPU
		// memory to accumulate intermediate results in the kernel.
		Dtype* loss_data = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0, loss_data);
		// Similarly, this memory is never used elsewhere, and thus we can use it
		// to avoid having to allocate additional GPU memory.
		Dtype* counts = prob_.mutable_gpu_diff();
		caffe_gpu_set(prob_.count(), (Dtype)0, counts);
		// NOLINT_NEXT_LINE(whitespace/operators)
		SoftmaxLossForwardGPU2<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, loss_data,
			outer_num_, dim, inner_num_, counts);

		Dtype* loss_data2 = bottom[0]->mutable_gpu_diff();
		Dtype* counts2 = prob_.mutable_gpu_diff();

		for (int n = 0; n < bottom[0]->count() / inner_num_; ++n)
		{
			Dtype* count = coeff_.mutable_cpu_data() + n;
			caffe_gpu_asum(inner_num_, counts2, count);
			if (*count != 0)
				caffe_gpu_scal(inner_num_, (Dtype)(1.0/(*count)), loss_data2);

			loss_data2 += inner_num_;
			counts2 += inner_num_;
		}

		Dtype loss;
		caffe_gpu_asum(bottom[0]->count(), loss_data, &loss);

		loss /= outer_num_;
		top[0]->mutable_cpu_data()[0] = loss;
		if (top.size() == 2) {
			top[1]->ShareData(prob_);
		}
	}

	template <typename Dtype>
	__global__ void SoftmaxLossBackwardGPU2(const int nthreads, const Dtype* top,
		const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
		const int spatial_dim) {
		const int channels = dim / spatial_dim;

		CUDA_KERNEL_LOOP(index, nthreads) {
			const int n = index / spatial_dim;
			const int s = index % spatial_dim;
			const int label_value = static_cast<int>(label[n * spatial_dim + s]);


			int idx = n * dim + label_value * spatial_dim + s;
			bottom_diff[idx] -= 1;

		}
	}
	template<typename Dtype>
	__global__ void normalize(const int nthreads, const Dtype* counts, const Dtype* label,
		const int spatial_dim, const int channels, Dtype* bottom_diff)
	{
		CUDA_KERNEL_LOOP(index, nthreads)
		{
			const int sp = index % spatial_dim;
			const int ch = index / spatial_dim % channels;
			const int n = index / spatial_dim / channels;

			const int label_value = static_cast<int>(label[n * spatial_dim + sp]);
			Dtype count = counts[n*channels + label_value];
			if (count != 0)
			{
				const Dtype coeff = 1.0 / count;
				bottom_diff[index] *= coeff;
			}
			else
			{
				bottom_diff[index] = 0;
			}
		}
	}

	template <typename Dtype>
	void SoftmaxWithLossRwLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) {
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const Dtype* prob_data = prob_.gpu_data();
			const Dtype* top_data = top[0]->gpu_data();
			caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);

			const Dtype* label = bottom[1]->gpu_data();
			const int dim = prob_.count() / outer_num_;
			const int nthreads = outer_num_ * inner_num_;
			// Since this memory is never used for anything else,
			// we use to to avoid allocating new GPU memory.
			const Dtype* counts = coeff_.gpu_data();
			// NOLINT_NEXT_LINE(whitespace/operators)

			SoftmaxLossBackwardGPU2<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
				CAFFE_CUDA_NUM_THREADS >> >(nthreads, top_data, label, bottom_diff,
				outer_num_, dim, inner_num_);
			const Dtype loss_weight = top[0]->cpu_diff()[0];

			const int count = bottom[0]->count();
			normalize<Dtype> << <CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS >> >(count, coeff_.gpu_data(), label, inner_num_,
				count / inner_num_ / outer_num_, bottom_diff);

			caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
		}
	}


INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossRwLayer);

}  // namespace caffe
