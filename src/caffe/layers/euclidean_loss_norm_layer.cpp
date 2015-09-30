#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void EuclideanLossNormLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
			<< "Inputs must have the same dimension.";
		diff_.ReshapeLike(*bottom[0]);
		diff2_.ReshapeLike(*bottom[0]);
		norm_.Reshape(bottom[0]->num(), 1, 1, 1);
		loss_.Reshape(bottom[0]->num(), 1, 1, 1);
	}

	template <typename Dtype>
	void EuclideanLossNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int count = bottom[0]->count();
		caffe_sub(
			count,
			bottom[0]->cpu_data(),
			bottom[1]->cpu_data(),
			diff_.mutable_cpu_data());
		caffe_sub(
			count,
			bottom[2]->cpu_data(),
			bottom[1]->cpu_data(),
			diff2_.mutable_cpu_data());

		int tmp_offset = bottom[0]->offset(1);
		Dtype* loss_data = loss_.mutable_cpu_data();
		Dtype* norm_data = norm_.mutable_cpu_data();

		for (int n = 0; n < bottom[0]->num(); ++n)
		{

			loss_data[n] = caffe_cpu_dot(tmp_offset, diff_.cpu_data() + diff_.offset(n),
				diff_.cpu_data() + diff_.offset(n));
			norm_data[n] = caffe_cpu_dot(tmp_offset, diff2_.cpu_data() + diff2_.offset(n),
				diff2_.cpu_data() + diff2_.offset(n));
			norm_data[n] = 24 * sqrt(1.0 / (norm_data[n] + std::numeric_limits<Dtype>::epsilon()));
		}
		Dtype dot;
		dot = caffe_cpu_dot(bottom[0]->num(), loss_.cpu_data(), norm_.cpu_data());

		Dtype loss = dot / bottom[0]->num() / Dtype(2);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	void EuclideanLossNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
				caffe_cpu_axpby(loss_.count(), alpha, norm_.cpu_data(), (Dtype)0, loss_.mutable_cpu_diff());

				int tmp_offset = bottom[i]->offset(1);
				const Dtype* loss_diff = loss_.cpu_diff();
				for (int n = 0; n < bottom[i]->num(); ++n)
				{

					caffe_cpu_axpby(
						tmp_offset,              // count
						loss_diff[n],                              // alpha
						diff_.cpu_data() + diff_.offset(n),                   // a
						Dtype(0),                           // beta
						bottom[i]->mutable_cpu_diff() + bottom[i]->offset(n));  // b
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(EuclideanLossNormLayer);
#endif

	INSTANTIATE_CLASS(EuclideanLossNormLayer);
	REGISTER_LAYER_CLASS(EuclideanLossNorm);

}  // namespace caffe
