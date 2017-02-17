#include "caffe/layers/cluster_centroid_dist_layer.hpp"
#include <vector>

namespace caffe{
	//template <typename Dtype>
	//__global__ void eucliean_vec_forward(const int n, const int num, const int num_cluster,
	//	const int feat_dim, const Dtype* bottom_data, const Dtype* centroid_data,
	//	Dtype* top_data)
	//{
	//	CUDA_KERNEL_LOOP(index, n) {
	//		int k_idx = n % num_cluster;
	//		int n_idx = n / num_cluster;

	//		const Dtype* bottom_ptr = bottom_data + n_idx*feat_dim;
	//		const Dtype* centroid_ptr = centroid_data + k_idx*feat_dim;
	//		Dtype* top_ptr = top_data + n;
	//		top_ptr[0] = 0;
	//		for (int i = 0; i < feat_dim; ++i)
	//		{
	//			top_ptr[0] += 0.5*(bottom_ptr[i] - centroid_ptr[i])*(bottom_ptr[i] - centroid_ptr[i]);
	//		}

	//	}
	//}
	
#define BLOCK_DIM 16

	// This kernel is optimized to ensure all global reads and writes are coalesced,
	// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
	// than the naive kernel below.  Note that the shared memory array is sized to 
	// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
	// so that bank conflicts do not occur when threads address the array column-wise.
	__global__ void transpose(float *odata, float *idata, int width, int height)
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
	void ClusterCentroidDistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{ 
		//caffe_gpu_mul(bottom[0]->count(), bottom[0]->gpu_data(), bottom[0]->gpu_data(), bottom_cache_.mutable_gpu_data());
		//caffe_gpu_gemm(
		//	CblasNoTrans,
		//	CblasNoTrans,
		//	bottom[0]->num(),
		//	1,
		//	centroid_dim_,
		//	(Dtype)1.0,
		//	bottom_cache_.gpu_data(),
		//	ones_.gpu_data(),
		//	(Dtype)0.0,
		//	bottom_cache_.mutable_gpu_diff()
		//	);
		//caffe_gpu_powx(bottom[0]->num(), bottom_cache_.gpu_diff(), (Dtype)0.5, bottom_cache_.mutable_gpu_diff());
		//caffe_gpu_gemm(
		//	CblasNoTrans, CblasNoTrans,
		//	bottom[0]->num(),
		//	centroid_dim_,
		//	1,
		//	(Dtype)1.0,
		//	bottom_cache_.gpu_diff(),
		//	ones_.gpu_data(),
		//	(Dtype)0.0,
		//	bottom_cache_.mutable_gpu_data()
		//	);
		//caffe_gpu_div(bottom[0]->count(), bottom[0]->gpu_data(), bottom_cache_.gpu_data(), bottom_cache_.mutable_gpu_diff());
		//if (use_T_)
		//	caffe_gpu_scal(bottom[0]->count(), (Dtype)T, bottom_cache_.mutable_gpu_diff());

		//const Dtype* bottom_data = bottom_cache_.gpu_diff(); //= bottom[0]->gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* centroid_data = this->blobs_[0]->gpu_data();

		if (is_sample_base_cls)
		{
			caffe_copy(bottom[0]->count(), bottom_data, this->blobs_[0]->mutable_gpu_data());
			for (int i = 0; i < top[1]->count(); ++i)
			{
				top[1]->mutable_cpu_data()[i] = i;
			}
		}
		else
		{
			if (!initialized_)
			{
				if (init_count_ >= this->blobs_[0]->count())
				{
					initialized_ = true;
					LOG(INFO) << "intial centroid complete.";
				}
				else
				{
					int count = min(bottom[0]->count(), this->blobs_[0]->count() - init_count_);
					caffe_copy(count, bottom_data, this->blobs_[0]->mutable_gpu_data() + init_count_);
					caffe_rng_gaussian<Dtype>(top[0]->count(), Dtype(0),
						Dtype(1), top[0]->mutable_cpu_data());
					init_count_ += count;
					LOG(INFO) << init_count_;
					return;
				}
			}
		}
		// square of data.
		caffe_gpu_mul(bottom[0]->count(),
			bottom_data,
			bottom_data,
			square_feat_.mutable_gpu_data());
		// sum along centroid_dim_
		caffe_gpu_gemm(
			CblasNoTrans,
			CblasNoTrans,
			num_samp_,
			//bottom[0]->num(),
			1,
			centroid_dim_,
			(Dtype)0.5, // / centroid_dim_,
			square_feat_.gpu_data(),
			ones_.gpu_data(),
			(Dtype)0.0,
			column_.mutable_gpu_data());
		// span along num_cluster_ dim
		caffe_gpu_gemm(
			CblasNoTrans,
			CblasNoTrans,
			//bottom[0]->num(),
			num_samp_,
			num_cluster_,
			1,
			(Dtype)1.0,
			column_.gpu_data(),
			ones_.gpu_data(),
			(Dtype)0.0,
			top_data
			);
		// dot product of centroid and feat
		caffe_gpu_gemm(
			CblasNoTrans,
			CblasTrans,
			//bottom[0]->num(),
			num_samp_,
			num_cluster_,
			centroid_dim_,
			(Dtype)-1.0, // / centroid_dim_,
			bottom_data,
			centroid_data,
			(Dtype)1.0,
			top_data);


		//square of centroid.
		caffe_gpu_mul(
			this->blobs_[0]->count(), 
			centroid_data, 
			centroid_data, 
			square_cluster_.mutable_gpu_data());

		//sum along centroid_dim_
		caffe_gpu_gemm(
			CblasNoTrans,
			CblasNoTrans, 
			num_cluster_, 
			1,
			centroid_dim_,
			(Dtype)1.0, // / centroid_dim_,
			square_cluster_.gpu_data(),
			ones_.gpu_data(),
			(Dtype)0.0,
			column_.mutable_gpu_data());
		//span along feat num
		caffe_gpu_gemm(CblasNoTrans,
			CblasNoTrans, 
			//bottom[0]->num(),
			num_samp_,
			num_cluster_, 
			1,
			(Dtype)0.5,
			ones_.gpu_data(), 
			column_.gpu_data(),
			(Dtype)1.0,
			top_data);//cache_cluster_.mutable_gpu_data());


		caffe_gpu_powx(top[0]->count(), top_data, (Dtype)0.5, top_data);
		caffe_gpu_scal(top[0]->count(), (Dtype)scale, top_data);
	}


	template <typename Dtype>
	void ClusterCentroidDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{ 
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff(); 
		Dtype* centroid_diff = this->blobs_[0]->mutable_gpu_diff();

		if (is_sample_base_cls)
		{
		}
		else{
			if (!initialized_)
			{
				caffe_gpu_set(bottom[0]->count(), (Dtype)0, bottom_diff);
				caffe_gpu_set(this->blobs_[0]->count(), (Dtype)0, centroid_diff);
				return;
			}
		}

		//const Dtype* top_diff = temp_diff_.gpu_data();
		caffe_gpu_div(top[0]->count(), top[0]->gpu_diff(), top[0]->gpu_data(), top_cache_.mutable_gpu_data());
		caffe_gpu_scal(top[0]->count(), (Dtype)(scale / 2.0), top_cache_.mutable_gpu_data());

		const Dtype* top_diff = top_cache_.gpu_data(); //top[0]->gpu_diff();
		//const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* centroid_data = this->blobs_[0]->gpu_data();
		const Dtype* top_data = top[0]->gpu_data();
		
		//const Dtype* bottom_data = bottom_cache_.gpu_diff(); //= bottom[0]->gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();

		//**************propagate diff to centroid***************
		// dot top_diff with feat data.
		if (this->param_propagate_down_[0]){
			caffe_gpu_gemm(
				CblasTrans,
				CblasNoTrans,
				num_cluster_,
				centroid_dim_,
				top[0]->num(),
				(Dtype)scale,// / centroid_dim_,
				top_diff,
				//bottom[0]->gpu_data(),
				bottom_data,
				(Dtype)0.0,
				square_cluster_.mutable_gpu_data()
				);

			// sum top_diff along num
			caffe_gpu_gemm(
				CblasNoTrans,
				CblasNoTrans,
				1,
				num_cluster_,
				top[0]->num(),
				(Dtype)scale, // / centroid_dim_,
				ones_.gpu_data(),
				top_diff,
				(Dtype)0.0,
				column_.mutable_gpu_data()
				);
			// expand top_diff along centroid_dim_.
			caffe_gpu_gemm(
				CblasNoTrans,
				CblasNoTrans,
				num_cluster_,
				centroid_dim_,
				1,
				(Dtype)1.0,
				column_.gpu_data(),
				ones_.gpu_data(),
				(Dtype)0.0,
				//centroid_diff
				square_cluster_.mutable_gpu_diff()
				);

			// multipy with centroid data
			//caffe_gpu_mul(this->blobs_[0]->count(), centroid_diff, centroid_data, centroid_diff);
			caffe_gpu_mul(this->blobs_[0]->count(), square_cluster_.mutable_gpu_diff(), centroid_data, square_cluster_.mutable_gpu_diff());
			// sum all diff
			caffe_gpu_sub(this->blobs_[0]->count(), square_cluster_.mutable_gpu_diff(), square_cluster_.gpu_data(), square_cluster_.mutable_gpu_diff());
			caffe_gpu_add(this->blobs_[0]->count(), square_cluster_.mutable_gpu_diff(), centroid_diff, centroid_diff);
		}

		if (propagate_down[0])
		{
			//**************propagate diff to feat data**************
			// dot top_diff with centroid data
			caffe_gpu_gemm(
				CblasNoTrans,
				//CblasTrans,
				CblasNoTrans,
				top[0]->num(),
				centroid_dim_,
				num_cluster_,
				(Dtype)scale, // / centroid_dim_,
				top_diff,
				centroid_data,
				(Dtype)0.0,
				square_feat_.mutable_gpu_data()
				);
			// sum top_diff along num_cluster_
			caffe_gpu_gemm(
				CblasNoTrans,
				CblasNoTrans,
				top[0]->num(),
				1,
				num_cluster_,
				(Dtype)scale,// / centroid_dim_,
				top_diff,
				ones_.gpu_data(),
				(Dtype)0.0,
				column_.mutable_gpu_data()
				);
			//expand top_diff along centroid_dim_.
			caffe_gpu_gemm(
				CblasNoTrans,
				CblasNoTrans,
				top[0]->num(),
				centroid_dim_,
				1,
				(Dtype)1.0,
				column_.gpu_data(),
				ones_.gpu_data(),
				(Dtype)0.0,
				bottom_diff
				);
			//multipy with centroid data.
			caffe_gpu_mul(bottom[0]->count(), bottom_diff, bottom_data/*bottom[0]->gpu_data()*/, bottom_diff);
			//sum all diff
			caffe_gpu_sub(bottom[0]->count(), bottom_diff, square_feat_.gpu_data(), bottom_diff);
		}

		//Dtype scalar_factor = (top[0]->asum_diff() / top[0]->count()) / (bottom[0]->asum_diff() / bottom[0]->count());//top_cache_.asum_data();
		//LOG(INFO) << "scale:" << scalar_factor << "," << bottom[0]->cpu_diff()[0] << "," << top[0]->cpu_diff()[0] << "," << top[0]->cpu_data()[0];
		//caffe_gpu_scal(top[0]->count(), (Dtype)scalar_factor, bottom[0]->mutable_gpu_diff());

		//caffe_gpu_div(bottom[0]->count(), bottom_diff, bottom_cache_.gpu_data(), bottom_diff);
		//caffe_gpu_mul(bottom[0]->count(), bottom_diff, bottom_cache_.gpu_diff(), bottom_cache_.mutable_gpu_data());
		//caffe_gpu_gemm(
		//	CblasNoTrans,
		//	CblasNoTrans,
		//	bottom[0]->num(),
		//	1,
		//	centroid_dim_,
		//	(Dtype)1.0,
		//	bottom_cache_.gpu_data(),
		//	ones_.gpu_data(),
		//	(Dtype)0.0,
		//	column_.mutable_gpu_data()
		//	);
		//caffe_gpu_gemm(
		//	CblasNoTrans,
		//	CblasNoTrans,
		//	bottom[0]->num(),
		//	centroid_dim_,
		//	1,
		//	(Dtype)1.0,
		//	column_.gpu_data(),
		//	ones_.gpu_data(),
		//	(Dtype)0.0, 
		//	bottom_cache_.mutable_gpu_data()
		//	);
		//caffe_gpu_mul(bottom[0]->count(), bottom_cache_.gpu_diff(), bottom_cache_.gpu_data(), bottom_cache_.mutable_gpu_data());
		//if (use_T_)
		//{
		//	caffe_gpu_scal(bottom[0]->count(), (Dtype)(1.0 / T), bottom_cache_.mutable_gpu_data());
		//	caffe_gpu_scal(bottom[0]->count(), (Dtype)( T), bottom_diff);
		//}
		//caffe_gpu_sub(bottom[0]->count(), bottom_diff, bottom_cache_.gpu_data(), bottom_diff);

		////////debug
		//for (int i = 0; i < 20; ++i)
		//	LOG(INFO) <<"#"<<i<<": " << this->blobs_[0]->cpu_data()[i] << "," << this->blobs_[0]->cpu_diff()[i] << ","
		//	<< bottom_cache_.cpu_diff()[i] <<"," <<bottom[0]->cpu_data()[i] << "," << bottom[0]->cpu_diff()[i] << "," << top[0]->cpu_data()[i] << "," << top[0]->cpu_diff()[i];

		//caffe_gpu_gemm(
		//	CblasNoTrans, 
		//	CblasNoTrans,
		//	top[0]->num(), 
		//	centroid_dim_, 
		//	num_cluster_,
		//	(Dtype)1.0*scale, 
		//	top_diff, 
		//	this->blobs_[1]->gpu_data(), 
		//	(Dtype)0.0, 
		//	square_feat_.mutable_gpu_data());
		////multipy feat data
		//caffe_gpu_mul(bottom[0]->count(), 
		//	square_feat_.gpu_data(), 
		//	bottom[0]->gpu_data(), 
		//	square_feat_.mutable_gpu_data());


		//caffe_gpu_mul(
		//	this->blobs_[0]->count(), 
		//	centroid_data, 
		//	this->blobs_[1]->gpu_data(), 
		//	square_cluster_.mutable_gpu_data());

		////dot diff of feat
		//caffe_gpu_gemm(
		//	CblasNoTrans, 
		//	CblasNoTrans, 
		//	top[0]->num(), 
		//	centroid_dim_, num_cluster_,
		//	(Dtype)-1.0*scale, 
		//	top_diff, 
		//	square_cluster_.gpu_data(), 
		//	(Dtype)0.0, bottom_diff);

		////dot diff of centroid //problem?
		//caffe_gpu_gemm(CblasTrans,
		//	CblasNoTrans, 
		//	num_cluster_, 
		//	centroid_dim_, 
		//	top[0]->num(),
		//	(Dtype)-1.0*scale, 
		//	top_diff, top_data, 
		//	(Dtype)0.0, 
		//	centroid_diff);

		//caffe_gpu_mul(this->blobs_[0]->count(),
		//	centroid_diff, 
		//	this->blobs_[1]->gpu_data(), 
		//	centroid_diff);


		////sum diff along feat num
		//caffe_gpu_gemm(CblasNoTrans, 
		//	CblasNoTrans, 
		//	1, 
		//	num_cluster_,
		//	top[0]->num(),
		//	(Dtype)1.0*scale, 
		//	ones_.gpu_data(), 
		//	top_diff,(Dtype)0.0, 
		//	column_.mutable_gpu_data());
		////span diff along centroid dim
		//caffe_gpu_gemm(CblasNoTrans, 
		//	CblasNoTrans,
		//	num_cluster_, 
		//	centroid_dim_, 
		//	1,
		//	(Dtype)1.0, 
		//	column_.gpu_data(), 
		//	ones_.gpu_data(),
		//	(Dtype)0.0, 
		//	square_cluster_.mutable_gpu_data());

		////multipy centroid data
		//caffe_gpu_mul(
		//	this->blobs_[0]->count(),
		//	square_cluster_.gpu_data(),
		//	this->blobs_[0]->gpu_data(), 
		//	square_cluster_.mutable_gpu_data());
		////multiply std normalizar.
		//caffe_gpu_mul(
		//	this->blobs_[0]->count(), 
		//	square_cluster_.gpu_data(), 
		//	this->blobs_[1]->gpu_data(), 
		//	square_cluster_.mutable_gpu_data());


		////sum all diff.
		//caffe_gpu_add(
		//	bottom[0]->count(),
		//	bottom_diff, 
		//	square_feat_.gpu_data(), 
		//	bottom_diff);
		//caffe_gpu_add(
		//	this->blobs_[0]->count(), 
		//	centroid_diff,
		//	square_cluster_.gpu_data(), 
		//	centroid_diff);

	}
	INSTANTIATE_LAYER_GPU_FUNCS(ClusterCentroidDistLayer);
}