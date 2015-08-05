#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
	template <typename Dtype>
	void KSparseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		KSparseParameter k_sparse_param = this->layer_param_.k_sparse_param();
		switch (k_sparse_param.sparse_type())
		{
		case KSparseParameter_SparseMethod_CHANNEL:
			CHECK_LE(k_sparse_param.sparse_k(), bottom[0]->channels())
				<< "sparse_k parameter must be less than number of bottom channels";
			break;
		case KSparseParameter_SparseMethod_SPAT:
			CHECK_LE(k_sparse_param.sparse_k(), bottom[0]->height()*bottom[0]->width())
				<< "sparse_k parameter must be less than number of bottom spatial pixels";
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}

		sparse_k = k_sparse_param.sparse_k();
		
	}

	template <typename Dtype>
	void KSparseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";

		top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());

		rank_val_.Reshape(bottom[0]->num(), sparse_k,
			bottom[0]->height(), bottom[0]->width());
		rank_idx_.Reshape(bottom[0]->num(), sparse_k, bottom[0]->height(),
			bottom[0]->width());

		if (top.size() > 1)
		{
			top[1]->Reshape(bottom[0]->num(), sparse_k, bottom[0]->height(), bottom[0]->width());
		}
	}

	template <typename Dtype>
	void KSparseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int top_count = top[0]->count();
		caffe_set(top_count, Dtype(0), top_data);

		Dtype* rank_val = NULL;
		int* rank_idx = NULL;
		Dtype* top_mask = NULL;

		int rank_count = rank_val_.count();
		int stride_ch_b = bottom[0]->offset(0, 1);
		int stride_ch_r = rank_val_.offset(0, 1);
		int stride_t = top[0]->offset(0, 1);
		switch (this->layer_param_.k_sparse_param().sparse_type())
		{
		case KSparseParameter_SparseMethod_CHANNEL:

			rank_val = rank_val_.mutable_cpu_data();
			rank_idx = rank_idx_.mutable_cpu_data();
			caffe_set(rank_count, Dtype(-FLT_MAX), rank_val);

			for (int n = 0; n < bottom[0]->num(); ++n)
			{
				for (int c = 0; c < bottom[0]->channels(); ++c)
				{
					for (int h = 0; h < bottom[0]->height(); ++h)
					{
						for (int w = 0; w < bottom[0]->width(); ++w)
						{
							int index_b = h*bottom[0]->width() + w;
							int cr;
							Dtype b_val = bottom_data[index_b];
							
							// search position and move
							for (cr = 0; cr < rank_val_.channels(); ++cr)
							{
								int index_r = cr*stride_ch_r + index_b;
								if (rank_val[index_r] >= b_val)
									break;
								if (cr + 1 < rank_val_.channels())
								{
									int index_r_last = (cr + 1)*stride_ch_r + index_b;
									rank_val[index_r] = rank_val[index_r_last];
									rank_idx[index_r] = rank_idx[index_r_last];
								}
							}

							// set b_val on cr-1
							if (cr - 1 > 0)
							{
								int index_r = (cr - 1)*stride_ch_r + index_b;
								rank_val[index_r] = b_val;
								rank_idx[index_r] = c;
							}
						}
					}
					bottom_data += bottom[0]->offset(0, 1);
				}
				rank_val += rank_val_.offset(1);
				rank_idx += rank_idx_.offset(1);
			}

			rank_idx = rank_idx_.mutable_cpu_data();
			rank_val = rank_val_.mutable_cpu_data();
			for (int n = 0; n < rank_idx_.num(); ++n)
			{
				for (int c = 0; c < rank_idx_.channels(); ++c)
				{
					for (int h = 0; h < rank_idx_.height(); ++h)
					{
						for (int w = 0; w < rank_idx_.width(); ++w)
						{
							int idx_r = h*rank_idx_.width() + w;
							int idx_t = rank_idx[idx_r] * stride_t + idx_r;
							top_data[idx_t] = rank_val[idx_r];
						}
					}
					rank_val += rank_val_.offset(0, 1);
					rank_idx += rank_idx_.offset(0, 1);
				}
				top_data += top[0]->offset(1);
			}

			if (top.size() > 1)
			{
				rank_idx = rank_idx_.mutable_cpu_data();
				top_mask = top[1]->mutable_cpu_data();
				for (int i = 0; i < rank_idx_.count(); i++)
					top_mask[i] = (Dtype)rank_idx[i];
			}
			break;
		case KSparseParameter_SparseMethod_SPAT:
			NOT_IMPLEMENTED;
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
	}

	template <typename Dtype>
	void KSparseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if (!propagate_down[0]) {
			return;
		}
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

		const int* rank_idx = NULL;
		int stride_t = top[0]->offset(0, 1);

		switch (this->layer_param_.k_sparse_param().sparse_type())
		{
		case KSparseParameter_SparseMethod_CHANNEL:
			rank_idx = rank_idx_.cpu_data();

			for (int n = 0; n < rank_idx_.num(); ++n)
			{
				for (int c = 0; c < rank_idx_.channels(); ++c)
				{
					for (int h = 0; h < rank_idx_.height(); ++h)
					{
						for (int w = 0; w < rank_idx_.width(); ++w)
						{
							int idx_r = h*rank_idx_.width() + w;
							int idx_t = rank_idx[idx_r] * stride_t + idx_r;
							bottom_diff[idx_t] += top_diff[idx_t];
						}
					}
					rank_idx += rank_idx_.offset(0, 1);
				}
				top_diff += top[0]->offset(1);
				bottom_diff += bottom[0]->offset(1);
			}
			break;
		case KSparseParameter_SparseMethod_SPAT:
			NOT_IMPLEMENTED;
			break;
		default:
			LOG(FATAL) << "Unknown pooling method.";
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(UnPoolingLayer);
#endif

	INSTANTIATE_CLASS(KSparseLayer);
	REGISTER_LAYER_CLASS(KSparse);
}