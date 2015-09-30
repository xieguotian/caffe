#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void EuclideanLossNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int count = bottom[0]->count();
		caffe_gpu_sub(
			count,
			bottom[0]->gpu_data(),
			bottom[1]->gpu_data(),
			diff_.mutable_gpu_data());
		caffe_gpu_sub(
			count,
			bottom[2]->gpu_data(),
			bottom[1]->gpu_data(),
			diff2_.mutable_gpu_data());
		Dtype dot;
		
		//Dtype dot_norm;
		Dtype* norm_data = norm_.mutable_cpu_data();
		int tmp_offset = bottom[0]->offset(1);
		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			caffe_gpu_dot(tmp_offset, diff_.gpu_data() + diff_.offset(n),
				diff_.gpu_data() + diff_.offset(n), loss_.mutable_cpu_data() + loss_.offset(n));

			caffe_gpu_dot(tmp_offset, diff2_.gpu_data() + diff2_.offset(n),
				diff2_.gpu_data() + diff2_.offset(n), norm_data + norm_.offset(n));
			norm_data[n] = 24 * sqrt(1.0 / (norm_data[n] + std::numeric_limits<Dtype>::epsilon()));
		}
		caffe_gpu_dot(bottom[0]->num(), loss_.gpu_data(), norm_.gpu_data(), &dot);

		Dtype loss = dot / bottom[0]->num() / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void EuclideanLossNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
				caffe_gpu_axpby(loss_.count(), alpha, norm_.gpu_data(), (Dtype)0, loss_.mutable_gpu_diff());
				int tmp_offset = bottom[i]->offset(1);
				const Dtype* loss_diff = loss_.cpu_diff();
				for (int n = 0; n < bottom[i]->num(); ++n)
				{
					
					caffe_gpu_axpby(
						tmp_offset,              // count
						loss_diff[n],                              // alpha
						diff_.gpu_data() + diff_.offset(n),                   // a
						Dtype(0),                           // beta
						bottom[i]->mutable_gpu_diff() + bottom[i]->offset(n));  // b
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossNormLayer);

}  // namespace caffe
