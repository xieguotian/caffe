#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe 
{
	template <typename Dtype>
	void NonLocalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		split_layer_0->Forward(bottom, split_0_top_vec);

		for (int n = 0; n < num_; ++n)
		{
			im2col_gpu(split_0_top_vec[0]->gpu_data() + split_0_top_vec[0]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				img2col_0_top.mutable_gpu_data() + img2col_0_top.offset(n));

			im2col_center_gpu(split_0_top_vec[1]->gpu_data() + split_0_top_vec[1]->offset(n),
				channels_, height_, width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				img2col_1_top.mutable_gpu_data() + img2col_1_top.offset(n));
		}

		split_layer_1->Forward(split_1_bottom_vec, split_1_top_vec);
		euclidean_bottom_vec[0]->ShareData(*split_1_top_vec[1]);
		euclidean_layer->Forward(euclidean_bottom_vec, euclidean_top_vec);

		caffe_gpu_scal(euclidean_top_vec[0]->count(),
			(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_gpu_data());

		smooth_threshold_layer->Forward(smooth_bottom_vec, smooth_top_vec);
		split_layer_3->Forward(split_3_bottom_vec, split_3_top_vec);
		normalize_bottom_vec[0]->ShareData(*split_3_top_vec[1]);
		normalize_layer->Forward(normalize_bottom_vec, normalize_top_vec);
		//top[1]->ShareData(*normalize_top_vec[0]);
		const Dtype* normalize_top_data = normalize_top_vec[0]->gpu_data();
		Dtype* top_1_data = top[1]->mutable_gpu_data();
		const int norm_offset = normalize_top_vec[0]->offset(1);

		for (int n = 0; n < normalize_top_vec[0]->num(); ++n)
		{
			for (int ch = 0; ch < channels_; ++ch)
			{
				caffe_copy(norm_offset, normalize_top_data, top_1_data);
				top_1_data += norm_offset;
			}
			normalize_top_data += norm_offset;
		}

		//int tmp_offset = smooth_top_vec[0]->count() / smooth_top_vec[0]->num();
		const int tmp_offset = split_3_top_vec[0]->offset(1);

		Dtype* split_2_bottom_data = split_2_bottom_vec[0]->mutable_gpu_data();
		//const Dtype* smooth_top_data = smooth_top_vec[0]->gpu_data();
		const Dtype* split_3_top_data = split_3_top_vec[0]->gpu_data();
		for (int n = 0; n < split_2_bottom_vec[0]->num(); ++n)
		{
			for (int ch = 0; ch < channels_; ++ch)
			{
				//caffe_copy(tmp_offset, smooth_top_data, split_2_bottom_data);
				caffe_copy(tmp_offset, split_3_top_data, split_2_bottom_data);
				split_2_bottom_data += tmp_offset;
			}
			//smooth_top_data += smooth_top_vec[0]->offset(1);
			split_3_top_data += tmp_offset;
		}

		split_layer_2->Forward(split_2_bottom_vec, split_2_top_vec);
		if (top.size() == 3)
			eltwise_layer->Forward(eltwise_bottom_vec, eltwise_top_vec);

	}

	template <typename Dtype>
	void NonLocalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		propagate_down_sub.push_back(propagate_down[0]);
		if (propagate_down[0])
		{
			for (int i = 0; i < eltwise_bottom_vec.size(); i++)
				caffe_gpu_set(eltwise_bottom_vec[i]->count(), (Dtype)0, eltwise_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < smooth_bottom_vec.size(); i++)
				caffe_gpu_set(smooth_bottom_vec[i]->count(), (Dtype)0, smooth_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < eltwise_bottom_vec.size(); i++)
				caffe_gpu_set(eltwise_bottom_vec[i]->count(), (Dtype)0, eltwise_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < split_1_bottom_vec.size(); i++)
				caffe_gpu_set(split_1_bottom_vec[i]->count(), (Dtype)0, split_1_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < smooth_top_vec.size(); i++)
				caffe_gpu_set(smooth_top_vec[i]->count(), (Dtype)0, smooth_top_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < split_0_top_vec.size(); i++)
				caffe_gpu_set(split_0_top_vec[i]->count(), (Dtype)0, split_0_top_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < split_3_top_vec.size(); i++)
				caffe_gpu_set(split_3_top_vec[i]->count(), (Dtype)0, split_3_top_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < normalize_top_vec.size(); i++)
				caffe_gpu_set(normalize_top_vec[i]->count(), (Dtype)0, normalize_top_vec[i]->mutable_gpu_diff());

			if (top.size() == 3)
				eltwise_layer->Backward(eltwise_top_vec, propagate_down_sub, eltwise_bottom_vec);

			split_layer_2->Backward(split_2_top_vec, propagate_down_sub, split_2_bottom_vec);
			//int tmp_offset = smooth_top_vec[0]->offset(1);
			const int tmp_offset = split_3_top_vec[0]->offset(1);
			//const Dtype* eltwise_bottom_1_diff = eltwise_bottom_vec[1]->gpu_diff();
			const Dtype* split_2_bottom_diff = split_2_bottom_vec[0]->gpu_diff();
			//Dtype* smooth_top_diff = smooth_top_vec[0]->mutable_gpu_diff();
			Dtype* split_3_top_diff = split_3_top_vec[0]->mutable_gpu_diff();
			for (int n = 0; n < split_2_bottom_vec[0]->num(); ++n)
			{
				for (int ch = 0; ch < channels_; ++ch)
				{
					//caffe_gpu_add(tmp_offset, smooth_top_diff, split_2_bottom_diff, smooth_top_diff);
					caffe_gpu_add(tmp_offset, split_3_top_diff, split_2_bottom_diff, split_3_top_diff);
					split_2_bottom_diff += tmp_offset;
				}
				//smooth_top_diff += tmp_offset;
				split_3_top_diff += tmp_offset;
			}

			const int norm_offset = normalize_top_vec[0]->offset(1);
			Dtype* normalize_diff = normalize_top_vec[0]->mutable_gpu_diff();
			const Dtype* top_1_diff = top[1]->gpu_diff();
			for (int n = 0; n < normalize_top_vec[0]->num(); ++n)
			{
				for (int ch = 0; ch < channels_; ++ch)
				{
					caffe_gpu_add(tmp_offset, normalize_diff, top_1_diff, normalize_diff);
					top_1_diff += norm_offset;
				}
				normalize_diff += norm_offset;
			}

			//normalize_top_vec[0]->ShareDiff(*top[1]);
			normalize_layer->Backward(normalize_top_vec, propagate_down_sub, normalize_bottom_vec);
			split_3_top_vec[1]->ShareDiff(*normalize_bottom_vec[0]);
			split_layer_3->Backward(split_3_top_vec, propagate_down_sub, split_3_bottom_vec);
			smooth_threshold_layer->Backward(smooth_top_vec, propagate_down_sub, smooth_bottom_vec);

			caffe_gpu_scal(euclidean_top_vec[0]->count(),
				(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_gpu_diff());

			euclidean_layer->Backward(euclidean_top_vec, propagate_down_sub, euclidean_bottom_vec);
			split_1_top_vec[1]->ShareDiff(*euclidean_bottom_vec[0]);
			split_layer_1->Backward(split_1_top_vec, propagate_down_sub, split_1_bottom_vec);

			for (int n = 0; n < num_; ++n)
			{
				col2im_center_gpu(img2col_1_top.gpu_diff() + img2col_1_top.offset(n), channels_, height_, width_,
					kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
					split_0_top_vec[1]->mutable_gpu_diff() + split_0_top_vec[1]->offset(n));

				col2im_gpu(img2col_0_top.gpu_diff() + img2col_0_top.offset(n), channels_, height_, width_,
					kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
					split_0_top_vec[0]->mutable_gpu_diff() + split_0_top_vec[0]->offset(n));
			}
			split_layer_0->Backward(split_0_top_vec, propagate_down_sub, bottom);

			CUDA_POST_KERNEL_CHECK;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NonLocalLayer);
}