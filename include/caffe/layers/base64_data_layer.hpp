#ifndef CAFFE_BASE64_DATA_LAYER_HPP_
#define CAFFE_BASE64_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Provides data to the Net from image files.
	*
	* TODO(dox): thorough documentation for Forward and proto params.
	*/
	template <typename Dtype>
	class Base64DataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit Base64DataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~Base64DataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Base64Data"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		shared_ptr<Caffe::RNG> prefetch_rng_;
		//virtual void ShuffleImages();
		virtual void load_batch(Batch<Dtype>* batch);
		virtual string base64_decode(string const& encoded_string);
		/*virtual void get_folder_file_range(string root_path);
		unsigned int num_files_;
		unsigned int current_file_;
		unsigned int first_file_;
		unsigned int last_file_;*/
		unsigned int lines_id_;
		std::ifstream infile;
		//string file_postfix_ = "_batch.tsv";
	};


}  // namespace caffe

#endif  // CAFFE_BASE64_DATA_LAYER_HPP_
