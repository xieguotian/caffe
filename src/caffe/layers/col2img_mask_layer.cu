#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/col2img_mask_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void Col2imgMaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		Dtype* mask_out_data = mask_out_.mutable_gpu_data();
		Dtype* data_out_data = data_out_.mutable_gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		split_layer->Forward(split_bottom_vec, split_top_vec);
		eltwise_layer->Forward(eltwise_bottom_vec, eltwise_top_vec);

		const Dtype* eltwise_top_data = eltwise_top_vec[0]->gpu_data();
		const Dtype* mask_in_data = split_top_vec[1]->gpu_data();

		for (int n = 0; n < bottom[0]->num(); ++n) {
			col2im_gpu(eltwise_top_data + eltwise_top_vec[0]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				data_out_data + data_out_.offset(n));
			col2im_gpu(mask_in_data + split_top_vec[1]->offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				mask_out_data + mask_out_.offset(n));
		}
		caffe_gpu_add_scalar(mask_out_.count(), (Dtype)std::numeric_limits<Dtype>::epsilon(), mask_out_data);
		caffe_gpu_div(top[0]->count(), data_out_data, mask_out_data, top_data);
	}

	template <typename Dtype>
	void Col2imgMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		for (int i = 0; i < split_top_vec.size(); ++i)
			caffe_gpu_set(split_top_vec[i]->count(), (Dtype)0, split_top_vec[i]->mutable_gpu_diff());
		for (int i = 0; i < eltwise_top_vec.size(); ++i)
			caffe_gpu_set(eltwise_top_vec[i]->count(), (Dtype)0, eltwise_top_vec[i]->mutable_gpu_diff());
		caffe_gpu_set(mask_out_.count(), (Dtype)0, mask_out_.mutable_gpu_diff());
		caffe_gpu_set(data_out_.count(), (Dtype)0, data_out_.mutable_gpu_diff());

		const Dtype* mask_out_data = mask_out_.gpu_data();
		const Dtype* data_out_data = data_out_.gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();

		Dtype* mask_out_diff = mask_out_.mutable_gpu_diff();
		Dtype* data_out_diff = data_out_.mutable_gpu_diff();
		Dtype* eltwise_top_diff = eltwise_top_vec[0]->mutable_gpu_diff();
		Dtype* mask_in_diff = split_top_vec[1]->mutable_gpu_diff();

		caffe_gpu_div(top[0]->count(), top_diff, mask_out_data, data_out_diff);
		caffe_gpu_mul(mask_out_.count(), mask_out_data, mask_out_data, mask_out_diff);
		caffe_gpu_div(mask_out_.count(), data_out_data, mask_out_diff, mask_out_diff);
		caffe_gpu_mul(mask_out_.count(), top_diff, mask_out_diff, mask_out_diff);
		caffe_gpu_scal(mask_out_.count(), (Dtype)-1.0, mask_out_diff);

		for (int n = 0; n < bottom[0]->num(); ++n)
		{
			im2col_gpu(data_out_diff + data_out_.offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				eltwise_top_diff + eltwise_top_vec[0]->offset(n));
			im2col_gpu(mask_out_diff + mask_out_.offset(n), channels_, height_, width_,
				kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
				mask_in_diff + split_top_vec[1]->offset(n));
		}

		vector<bool> propagate_down_sub;
		propagate_down_sub.push_back(propagate_down[0]);
		if (bottom.size() == 2)
			propagate_down_sub.push_back(propagate_down[1]);

		eltwise_layer->Backward(eltwise_top_vec, propagate_down_sub, eltwise_bottom_vec);
		split_layer->Backward(split_top_vec, propagate_down_sub, split_bottom_vec);
	}


	INSTANTIATE_LAYER_GPU_FUNCS(Col2imgMaskLayer);

}  // namespace caffe
