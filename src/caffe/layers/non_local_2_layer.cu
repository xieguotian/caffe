#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe 
{
	template <typename Dtype>
	void NonLocal2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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

		euclidean_bottom_vec[0]->ShareData(img2col_0_top);
		euclidean_bottom_vec[1]->ShareData(img2col_1_top);
		euclidean_layer->Forward(euclidean_bottom_vec, euclidean_top_vec);

		caffe_gpu_scal(euclidean_top_vec[0]->count(),
			(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_gpu_data());

		exp_layer->Forward(exp_bottom_vec, exp_top_vec);
		normalize_bottom_vec[0]->ShareData(*exp_top_vec[0]);
		normalize_layer->Forward(normalize_bottom_vec, normalize_top_vec);
	}

	template <typename Dtype>
	void NonLocal2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		propagate_down_sub.push_back(propagate_down[0]);
		if (propagate_down[0])
		{
			for (int i = 0; i < exp_bottom_vec.size(); i++)
				caffe_gpu_set(exp_bottom_vec[i]->count(), (Dtype)0, exp_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < euclidean_bottom_vec.size(); i++)
				caffe_gpu_set(euclidean_bottom_vec[i]->count(), (Dtype)0, euclidean_bottom_vec[i]->mutable_gpu_diff());
			for (int i = 0; i < split_0_top_vec.size(); i++)
				caffe_gpu_set(split_0_top_vec[i]->count(), (Dtype)0, split_0_top_vec[i]->mutable_gpu_diff());

			
			normalize_layer->Backward(normalize_top_vec, propagate_down_sub, normalize_bottom_vec);
			exp_top_vec[0]->ShareDiff(*normalize_bottom_vec[0]);
			exp_layer->Backward(exp_top_vec, propagate_down_sub, exp_bottom_vec);

			caffe_gpu_scal(euclidean_top_vec[0]->count(),
				(Dtype)(1.0 / bottom[0]->channels()), euclidean_top_vec[0]->mutable_gpu_diff());

			euclidean_layer->Backward(euclidean_top_vec, propagate_down_sub, euclidean_bottom_vec);
			img2col_0_top.ShareDiff(*euclidean_bottom_vec[0]);
			img2col_1_top.ShareDiff(*euclidean_bottom_vec[1]);

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

		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NonLocal2Layer);
}