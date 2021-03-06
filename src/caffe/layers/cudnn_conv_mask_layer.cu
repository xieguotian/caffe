#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_conv_mask_layer.hpp"

namespace caffe {

__global__ void sync_conv_groups_t() { }

template <typename Dtype>
__global__ void max_among_six_spatial(const int nthreads,
	const Dtype* const input_data, 
	const int num, const int channels,
	const int height, const int width,
	Dtype* const output_data,unsigned char* const output_mask,Dtype factor)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w_idx = index % width;
		const int h_idx = (index / width) % height;
		const int ch_idx = (index / height / width) % channels;
		const int num_idx = index / channels / height / width;

		Dtype d[9];

		Dtype max_value = -FLT_MAX;
		int max_pos = -1;
		int g_idx;
		int tmp_w_idx;
		int tmp_h_idx;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				g_idx = i * 3 + j;
				tmp_w_idx = j - 1 + w_idx;
				tmp_h_idx = i - 1 + h_idx;
				if (tmp_w_idx < 0 || tmp_w_idx >= width || tmp_h_idx < 0 || tmp_h_idx >= height)
					d[g_idx] = 0;
				else
					d[g_idx] = input_data[(((num_idx*9+g_idx)*channels+ch_idx)*height+tmp_h_idx)*width+tmp_w_idx];
			}
		}
		
		//Dtype val[6];

		//val[0] = d[4];
		//val[1] = d[3] + d[4] + d[5];
		//val[2] = d[1] + d[4] + d[7];
		//val[3] = d[0] + d[4] + d[8];
		//val[4] = d[2] + d[4] + d[6];
		//val[5] = d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] + d[8];
		//max_value = d[0] + d[1] + d[2] + d[3];

		d[0] = (d[0] + d[4] + d[8]);
		d[1] = (d[1] + d[4] + d[7]);
		d[2] = (d[2] + d[4] + d[6]);
		d[3] = (d[3] + d[4] + d[5]);

		//d[5] = max_value + d[4] + d[5] + d[6] + d[7] + d[8];
		d[5] = (d[0] + d[1] + d[2] + d[3] - 3 * d[4]);
		//d[6] = d[5] - d[4];

		//max_value = val[0];
		max_value = d[0];
		max_pos = 0;
		//max_value = d[4];
		//max_pos = 4;
		//d[0] *= 0.003 * factor; //1.25;
		//d[1] *= 0.003 * factor; //1.25;
		//d[2] *= 0.003 * factor; //1.25;
		//d[3] *= 0.003 * factor; //1.25;
		//d[4] *= 0.009 * factor; //2;
		for (int i = 1; i < 6; i++)
		//for (int i = 1; i < 7; i++)
		//for (int i = 5; i < 6; i++)
		{
			//if (max_value < val[i])
			if (max_value < d[i])
			{
				//max_value = val[i];
				max_value = d[i];
				max_pos = i;
			}
		}
		//max_pos = 5;
		output_data[index] = d[max_pos];//max_value;
		output_mask[index] = (unsigned char)max_pos;
		//output_data[index] = d[5];
		//output_mask[index] = (unsigned char)5;

	}
}

template <typename Dtype>
__global__ void spatial_relu(const int nthreads,
	const Dtype* const input_data,
	const int num, const int channels,
	const int height, const int width,
	Dtype* const output_data, unsigned char* const output_mask, Dtype factor, Dtype leaky_factor=0)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w_idx = index % width;
		const int h_idx = (index / width) % height;
		const int ch_idx = (index / height / width) % channels;
		const int num_idx = index / channels / height / width;

		Dtype d[9];

		Dtype max_value = -FLT_MAX;
		int max_pos = -1;
		int g_idx;
		int tmp_w_idx;
		int tmp_h_idx;
		
		unsigned char spatial_record = 0;
		Dtype result = 0;
		Dtype tmp_data = 0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				g_idx = i * 3 + j;
				tmp_w_idx = j - 1 + w_idx;
				tmp_h_idx = i - 1 + h_idx;
				if (tmp_w_idx < 0 || tmp_w_idx >= width || tmp_h_idx < 0 || tmp_h_idx >= height)
					//d[g_idx] = 0;
					tmp_data = 0;
				else
					//d[g_idx] = input_data[(((num_idx * 9 + g_idx)*channels + ch_idx)*height + tmp_h_idx)*width + tmp_w_idx];
					tmp_data = input_data[(((num_idx * 9 + g_idx)*channels + ch_idx)*height + tmp_h_idx)*width + tmp_w_idx];

				if (g_idx == 4)
				{
					result += tmp_data;
				}
				else
				{
					bool is_not_zero = tmp_data > 0;
					if (is_not_zero)
					{
						result += tmp_data;

						spatial_record = (spatial_record << 1) | 1;
					}
					else
					{
						result += leaky_factor * tmp_data;
						spatial_record = (spatial_record << 1);
					}
				}
			}
		}

		output_data[index] = result;
		output_mask[index] = (unsigned char)spatial_record;
		//output_data[index] = d[5];
		//output_mask[index] = (unsigned char)5;

	}
}
template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();

  shared_ptr<Blob<Dtype>> caches_;
  thread_id_ = boost::lexical_cast<std::string>(boost::this_thread::get_id());
  caches_ = thread_caches_[thread_id_];
  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();

	Dtype* top_data = caches_->mutable_gpu_data(); //top[i]->mutable_gpu_data(); 
	// Forward through cuDNN in parallel over groups.
	for (int g = 0; g < this->group_; g++) {
		// Filters.
		CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
			cudnn::dataType<Dtype>::one,
			bottom_descs_[i], bottom_data + bottom_offset_ * g,
			filter_desc_, weight + this->weight_offset_ * g,
			conv_descs_[i],
			fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
			cudnn::dataType<Dtype>::zero,
			top_descs_[i], top_data + top_offset_ * g));

		// Bias.
		if (this->bias_term_) {
			const Dtype* bias_data = this->blobs_[1]->gpu_data();
			CUDNN_CHECK(cudnnAddTensor(handle_[g],
				cudnn::dataType<Dtype>::one,
				bias_desc_, bias_data + bias_offset_ * g,
				cudnn::dataType<Dtype>::one,
				top_descs_[i], top_data + top_offset_ * g));
		}
	}

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups_t<<<1, 1>>>();

	// forward data from cacehs to top
	int n_threads = top[i]->count();
	//Dtype* mask_data = top[i * 2 + 1]->mutable_gpu_data();
	unsigned char* mask_data = mask_caches_[i]->mutable_gpu_data();
	if (this->layer_param_.conv_mask_param().use_spatial())
	{
		Dtype leaky_factor = this->layer_param_.relu_param().negative_slope();
		static bool first_log = true;
		if (first_log)
		{
			first_log = false;
			LOG(INFO) << "use spatial_reul forward. negative_slope: " << leaky_factor;
		}
		spatial_relu<Dtype> << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
		n_threads, top_data, top[i]->num(), top[i]->channels(), top[i]->height(), top[i]->width(),
		top[i]->mutable_gpu_data(), mask_data, factor_, leaky_factor
			);
	}
	else
	{
		max_among_six_spatial<Dtype> << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			//spatial_relu<Dtype> << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, top_data, top[i]->num(), top[i]->channels(), top[i]->height(), top[i]->width(),
			top[i]->mutable_gpu_data(), mask_data, factor_
			);
	}
  }

}

template <typename Dtype>
__global__ void max_among_six_spatial_bp(const int nthreads,
	const Dtype* const input_diff,
	const int num, const int channels,
	const int height, const int width,
	Dtype* const output_diff, const unsigned char* const output_mask , Dtype factor)
{

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w_idx = index % width ;
		const int h_idx = (index / width) % height;
		const int ch_idx = (index / height / width) % channels;
		//const int g_idx = (index / height / width / channels) % 9;
		const int num_idx = index / channels / height / width;// / 9;

		//int j = g_idx % 3;
		//int i = g_idx / 3;
		Dtype diff_data = input_diff[index];
		int sel_num = output_mask[index];
		int g_idx_set[9];
		int num_g_idx = 0;
		Dtype norm_factor = 1.0;
		switch (sel_num)
		{
		case 4:
			g_idx_set[0] = 4;
			num_g_idx = 1;
			norm_factor = 0.009 * factor;//2.0;
			break;
		case 3:
			g_idx_set[0] = 3;
			g_idx_set[1] = 4;
			g_idx_set[2] = 5;
			num_g_idx = 3;
			norm_factor = 0.003 * factor;//1.25;
			break;
		case 1:
			g_idx_set[0] = 1;
			g_idx_set[1] = 4;
			g_idx_set[2] = 7;
			num_g_idx = 3;
			norm_factor = 0.003 * factor;//1.25;
			break;
		case 0:
			g_idx_set[0] = 0;
			g_idx_set[1] = 4;
			g_idx_set[2] = 8;
			num_g_idx = 3;
			norm_factor = 0.003 * factor;//1.25;
			break;
		case 2:
			g_idx_set[0] = 2;
			g_idx_set[1] = 4;
			g_idx_set[2] = 6;
			num_g_idx = 3;
			norm_factor = 0.003 * factor;//1.25;
			break;
		case 5:
			for (int i = 0; i < 9; i++)
				g_idx_set[i] = i;
			num_g_idx = 9;
			norm_factor = 1.0;
			break;
		case 6:
			int count = 0;
			for (int i = 0; i < 9; i++)
			{
				if (i != 4)
				{
					g_idx_set[count] = i;
					count++;
				}
			}
			num_g_idx = 8;
			norm_factor = 1.0;
			break;
		default:
			break;
		}
		for (int i = 0; i < num_g_idx; i++)
		{
			int g_idx = g_idx_set[i];
			int w = w_idx - 1 + (int)(g_idx % 3);
			int h = h_idx - 1 + (int)(g_idx / 3);
			if (w >= 0 && w < width && h >= 0 && h < height)
			{
				int index_t = (((num_idx * 9 + g_idx)*channels + ch_idx)*height + h)*width + w;
				output_diff[index_t] = diff_data;// *norm_factor;
			}
		}

		/*
		for (int g_idx = 0; g_idx < 9; g_idx++)
		{
			int index_t = (((num_idx * 9 + g_idx)*channels + ch_idx)*height + h_idx)*width + w_idx;
			//int w = w_idx - (int)(g_idx % 3) + 1;//j;
			//int h = h_idx - (int)(g_idx / 3) + 1;//i;

			int w = w_idx - 1 + (int)(g_idx % 3);
			int h = h_idx - 1 + (int)(g_idx / 3);

			if (w >= 0 && w < width && h >= 0 && h < height)
			{
				//int idx = ((num_idx*channels + ch_idx)*height + h)*width + w;
				//int sel_num = output_mask[idx];
				switch (sel_num)
				{
					//case 0:
				case 4:
					if (g_idx == 4)
						output_diff[index_t] = input_diff[idx];
					break;
					//case 1:
				case 3:
					if (g_idx == 3 || g_idx == 4 || g_idx == 5)
						output_diff[index_t] = input_diff[idx];
					break;
					//case 2:
				case 1:
					if (g_idx == 1 || g_idx == 4 || g_idx == 7)
						output_diff[index_t] = input_diff[idx];
					break;
					//case 3:
				case 0:
					if (g_idx == 0 || g_idx == 4 || g_idx == 8)
						output_diff[index_t] = input_diff[idx];
					break;
					//case 4:
				case 2:
					if (g_idx == 2 || g_idx == 4 || g_idx == 6)
						output_diff[index_t] = input_diff[idx];
					break;
				case 5:
					output_diff[index_t] = input_diff[idx];
					break;
				}
			}
		}
		*/

		//Dtype val = 0;
		//for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
		//	for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
		//		int idx = ((num_idx*channels + ch_idx)*height + h_col)*width + w_col;
		//		int sel_num = output_mask[idx];
		//		switch (sel_num)
		//		{
		//		case 0:
		//			if (g_idx == 4)
		//				val += input_diff[idx];
		//			break;
		//		case 1:
		//			if (g_idx == 3 || g_idx == 4 || g_idx == 5)
		//				val += input_diff[idx];
		//			break;
		//		case 2:
		//			if (g_idx == 1 || g_idx == 4 || g_idx == 7)
		//				val += input_diff[idx];
		//			break;
		//		case 3:
		//			if (g_idx == 0 || g_idx == 4 || g_idx == 8)
		//				val += input_diff[idx];
		//			break;
		//		case 4:
		//			if (g_idx == 2 || g_idx == 4 || g_idx == 6)
		//				val += input_diff[idx];
		//			break;
		//		case 5:
		//			val += input_diff[idx];
		//			break;
		//		}
		//	}
		//}
		//output_diff[index] = val;

		//Dtype val = 0;
		//const int w_im = index % width + 1;
		//const int h_im = (index / width) % height + 1;
		//const int c_im = index / (width * height);
		//const int n_idx = index / (width*height*channels * 9);
		//
		//int kernel_extent_w = 3;//kernel_w;
		//int kernel_extent_h = 3;// kernel_h;
		//// compute the start and end of the output
		//const int w_col_start =
		//	(w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) + 1;
		//const int w_col_end = min(w_im  + 1, width_col);
		//const int h_col_start =
		//	(h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) + 1;
		//const int h_col_end = min(h_im  + 1, height_col);

		//// TODO: use LCM of stride and dilation to avoid unnecessary loops
		//for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
		//	for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
		//		int h_k = (h_im - h_col);
		//		int w_k = (w_im - w_col);
		//		//int data_col_index = (((c_im * 3 + h_k) * 3 + w_k) *
		//		//	height_col + h_col) * width_col + w_col;
		//		int data_col_index = 
		//		val += input_diff[data_col_index];
		//	}
		//}
		//output_diff[index] = val;

		//const int w_idx = index % width;
		//const int h_idx = (index / width) % height;
		//const int ch_idx = (index / height / width) % channels;
		//const int num_idx = index / channels / height / width;
		//int sel_idx[9];
		//int num_sel = 0;
		//switch (output_mask[index])
		//{
		//case 0:
		//	sel_idx[0] = 4;
		//	num_sel = 1;
		//	break;
		//case 1:
		//	//int sel_idx[3] = { 3, 4, 5 };
		//	sel_idx[0] = 3; sel_idx[1] = 4; sel_idx[2] = 5;
		//	num_sel = 3;
		//	break;
		//case 2:
		//	//int sel_idx[3] = { 1, 4, 7 };
		//	sel_idx[0] = 1; sel_idx[1] = 4; sel_idx[2] = 7;
		//	num_sel = 3;
		//	break;
		//case 3:
		//	//int sel_idx[3] = { 0, 4, 8 };
		//	sel_idx[0] = 0; sel_idx[1] = 4; sel_idx[2] = 8;
		//	num_sel = 3;
		//	break;
		//case 4:
		//	//int sel_idx[3] = { 2, 4, 6 };
		//	sel_idx[0] = 2; sel_idx[1] = 4; sel_idx[2] = 6;
		//	num_sel = 3;
		//	break;
		//case 5:
		//	//int sel_idx[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
		//	for (int i = 0; i < 9; i++)
		//		sel_idx[i] = i;
		//	num_sel = 9;
		//	break;
		//}
		//for (int i = 0; i < 9; i++)
		//{
		//	int g_idx = i;
		//	int tmp_w_idx = int(g_idx % 3) - 1 + w_idx;
		//	int tmp_h_idx = int(g_idx / 3) - 1 + h_idx;
		//	if (tmp_w_idx < 0 || tmp_w_idx >= width || tmp_h_idx < 0 || tmp_h_idx >= height)
		//		continue;
		//	output_diff[(((num_idx * 9 + g_idx)*channels + ch_idx)*height + tmp_h_idx)*width + tmp_w_idx] = 0.1*input_diff[index];
		//}
		//for (int i = 0; i < num_sel; i++)
		//{
		//	int g_idx = sel_idx[i];
		//	int tmp_w_idx = int(g_idx % 3) - 1 + w_idx;
		//	int tmp_h_idx = int(g_idx / 3) - 1 + h_idx;
		//	if (tmp_w_idx < 0 || tmp_w_idx >= width || tmp_h_idx < 0 || tmp_h_idx >= height)
		//		continue;
		//	output_diff[(((num_idx * 9 + g_idx)*channels + ch_idx)*height + tmp_h_idx)*width + tmp_w_idx] = input_diff[index];
		//}
	}
}

template <typename Dtype>
__global__ void spatial_relu_bp(const int nthreads,
	const Dtype* const input_diff,
	const int num, const int channels,
	const int height, const int width,
	Dtype* const output_diff, const unsigned char* const output_mask, Dtype factor, Dtype leaky_factor=0)
{

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w_idx = index % width;
		const int h_idx = (index / width) % height;
		const int ch_idx = (index / height / width) % channels;
		//const int g_idx = (index / height / width / channels) % 9;
		const int num_idx = index / channels / height / width;// / 9;

		//int j = g_idx % 3;
		//int i = g_idx / 3;
		Dtype diff_data = input_diff[index];
		unsigned char sel_num = output_mask[index];
		Dtype norm_factor = 1.0;

		for (int i = 8; i >=0; i--)
		{
			int w = w_idx - 1 + (int)(i % 3);
			int h = h_idx - 1 + (int)(i / 3);

			if (i == 4)
			{
				if (w >= 0 && w < width && h >= 0 && h < height)
				{
					int index_t = (((num_idx * 9 + i)*channels + ch_idx)*height + h)*width + w;
					output_diff[index_t] = diff_data;// *norm_factor;
				}
			}
			else
			{
				bool is_not_zero = (sel_num & 1)>0;
				sel_num = sel_num >> 1;
				if (is_not_zero && w >= 0 && w < width && h >= 0 && h < height)
				{
					int index_t = (((num_idx * 9 + i)*channels + ch_idx)*height + h)*width + w;
					output_diff[index_t] = diff_data;// *norm_factor;
				}
				else if (w >= 0 && w < width && h >= 0 && h < height)
				{
					int index_t = (((num_idx * 9 + i)*channels + ch_idx)*height + h)*width + w;
					output_diff[index_t] = leaky_factor * diff_data;// *norm_factor;
				}
			}
		}

	}
}

template <typename Dtype>
__global__ void max_among_six_spatial_bp_fast(const int nthreads,
	const Dtype* const input_diff,
	const int num, const int channels,
	const int height, const int width,
	Dtype* const output_diff, const char* const output_mask)
{

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int g_idx = index % 9;
		const int w_idx = (index / 9)  % width;
		const int h_idx = (index / 9 / width) % height;
		const int ch_idx = (index / 9 / height / width) % channels;
		int w = w_idx - 1 + (int)(g_idx % 3);
		int h = h_idx - 1 + (int)(g_idx / 3); 
		const int num_idx = index / channels / height / width / 9;
		int out_idx = (((num_idx * 9 + g_idx)*channels + ch_idx)*height + h)*width + w; 
		 
		__shared__ float diff_data[50];
		__shared__ int idx_num[50];

		int diff_idx = threadIdx.x / 9;
		if (g_idx == 0)
		{
			int idx = ((num_idx*channels + ch_idx)*height + h_idx)*width + w_idx;
			diff_data[diff_idx] = input_diff[idx];
			idx_num[diff_idx] = output_mask[idx];
		}

		__syncthreads();

		if (w >= 0 && w < width && h >= 0 && h < height)
		{
			int sel_num = (int)idx_num[diff_idx];
			switch (sel_num)
			{
				//case 0:
			case 4:
				if (g_idx == 4)
					output_diff[out_idx] = (Dtype)diff_data[diff_idx];
					//output_diff[out_idx] = input_diff[idx];
				break;
				//case 1:
			case 3:
				if (g_idx == 3 || g_idx == 4 || g_idx == 5)
					output_diff[out_idx] = (Dtype)diff_data[diff_idx];
					//output_diff[out_idx] = input_diff[idx];
				break;
				//case 2:
			case 1:
				if (g_idx == 1 || g_idx == 4 || g_idx == 7)
					output_diff[out_idx] = (Dtype)diff_data[diff_idx];
					//output_diff[out_idx] = input_diff[idx];
				break;
				//case 3:
			case 0:
				if (g_idx == 0 || g_idx == 4 || g_idx == 8)
					output_diff[out_idx] = (Dtype)diff_data[diff_idx];
					//output_diff[out_idx] = input_diff[idx];
				break;
				//case 4:
			case 2:
				if (g_idx == 2 || g_idx == 4 || g_idx == 6)
					output_diff[out_idx] = diff_data[diff_idx];
					//output_diff[out_idx] = input_diff[idx];
				break;
			case 5:
				output_diff[out_idx] = (Dtype)diff_data[diff_idx];
				//output_diff[out_idx] = input_diff[idx]; 
				break;
			default:
				//output_diff[out_idx] = 0;
				break;
			}
		}
	}
}


template <typename Dtype>
__global__ void mask_diff(const int nthreads,
	const Dtype* const input_data,
	Dtype* const output_diff, Dtype mv_fr,
	const int num, const int channels,
	const int height, const int width)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w_idx = index % width;
		const int h_idx = index / width % height;
		const int n_idx = index / width / height; 

		Dtype sum = -1;

		for (int ch_idx = 0; ch_idx < channels; ch_idx++)
		{
			int idx = (((n_idx*channels + ch_idx)*height + h_idx)*width + w_idx);
			//sum += input_data[idx];
			if (sum<input_data[idx])
				sum = input_data[idx];
		}

		Dtype threshold = sum*mv_fr;

		for (int ch_idx = 0; ch_idx < channels; ch_idx++)
		{
			int idx = (((n_idx*channels + ch_idx)*height + h_idx)*width + w_idx);
			if (input_data[idx] <= threshold)
				output_diff[idx] = 0;
		}
	}
}

template <typename Dtype>
void CuDNNConvolutionMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  shared_ptr<Blob<Dtype>> caches_;
  thread_id_ = boost::lexical_cast<std::string>(boost::this_thread::get_id());
  caches_ = thread_caches_[thread_id_];

  for (int i = 0; i < top.size(); ++i) {

	  // propagate diff to caches.
	  int n_threads = top[i]->count() * 9;
	  caffe_gpu_set(n_threads, (Dtype)0, caches_->mutable_gpu_data());
	  const unsigned char* mask_data = mask_caches_[i]->gpu_data();
	  if (this->layer_param_.conv_mask_param().use_spatial())
	  {
		  Dtype leaky_factor = this->layer_param_.relu_param().negative_slope();
		  static bool first_log = true;
		  if (first_log)
		  {
			  first_log = false;
			  LOG(INFO) << "use spatial_reul backward. negative_slope: " << leaky_factor;
		  }
		  spatial_relu_bp<Dtype> << <CAFFE_GET_BLOCKS(n_threads / 9), CAFFE_CUDA_NUM_THREADS >> >(
			  n_threads / 9, top[i]->gpu_diff(), top[i]->num(),
			  top[i]->channels(), top[i]->height(), top[i]->width(),
			  caches_->mutable_gpu_data(), mask_data, factor_, leaky_factor);
	  }
	  else
	  {
		  max_among_six_spatial_bp<Dtype> << <CAFFE_GET_BLOCKS(n_threads / 9), CAFFE_CUDA_NUM_THREADS >> >(
			  //spatial_relu_bp<Dtype> << <CAFFE_GET_BLOCKS(n_threads / 9), CAFFE_CUDA_NUM_THREADS >> >(
			  n_threads / 9, top[i]->gpu_diff(), top[i]->num(),
			  top[i]->channels(), top[i]->height(), top[i]->width(),
			  caches_->mutable_gpu_data(), mask_data, factor_);
	  }
	  //int block_threads = (n_threads + 450 - 1) / 450;
	  ////int block_threads = (n_threads + 512 - 1) / 512;
	  //max_among_six_spatial_bp_fast<Dtype> <<<block_threads, 450>>>(
		 // n_threads, top[i]->gpu_diff(), top[i]->num(),
		 // top[i]->channels(), top[i]->height(), top[i]->width(),
		 // caches_->mutable_gpu_data(), mask_data);
	  const Dtype* top_diff = caches_->gpu_data();//top[i]->gpu_diff();

    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups_t<<<1, 1>>>();
  }

  //factor_ += 1;
  //if (factor_ > 1000)
	 // factor_ = 1000;
  //if (this->phase_ == TEST)
  //{
	 // Dtype mv_fr = this->layer_param_.batch_norm_param().moving_average_fraction();
	 // int n_threads = bottom[0]->count() / bottom[0]->channels();

	 // mask_diff << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
		//  n_threads, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff(),
		//  mv_fr,
		//  bottom[0]->num(), bottom[0]->channels(),
		//  bottom[0]->height(), bottom[0]->width()
		//  );
  //}
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionMaskLayer);

}  // namespace caffe
#endif
