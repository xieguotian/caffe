#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/base64_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
	static const std::string base64_chars = 
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

	static inline bool is_base64(unsigned char c) {
	  return (isalnum(c) || (c == '+') || (c == '/'));
	}

	template <typename Dtype>
	string Base64DataLayer<Dtype>::base64_decode(string const& encoded_string) {
		int in_len = encoded_string.size();
		int i = 0;
		int j = 0;
		int in_ = 0;
		unsigned char char_array_4[4], char_array_3[3];
		std::string ret;

		while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
			char_array_4[i++] = encoded_string[in_]; in_++;
			if (i == 4) {
				for (i = 0; i < 4; i++)
					char_array_4[i] = base64_chars.find(char_array_4[i]);

				char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
				char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
				char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

				for (i = 0; (i < 3); i++)
					ret += char_array_3[i];
				i = 0;
			}
		}

		if (i) {
			for (j = i; j < 4; j++)
				char_array_4[j] = 0;

			for (j = 0; j < 4; j++)
				char_array_4[j] = base64_chars.find(char_array_4[j]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

			for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
		}

		return ret;
	}

	template <typename Dtype>
	Base64DataLayer<Dtype>::~Base64DataLayer<Dtype>() {
		this->StopInternalThread();
	}

	/*
	template <typename Dtype>
	void Base64DataLayer<Dtype>::get_folder_file_range(string &root_path) {
		_finddata_t fileDir;
		first_file_ = 0;
		last_file_ = 0;
		num_files_ = 0;
		string path_temp;
		long lfDir;
		if ((lfDir = _findfirst(path_temp.assign(path).append("\*").c_str(), &fileDir)) != -1l){
			if (fileDir.name.find(file_postfix_) != -1){
				num_files++;
			}
		}
	}
	*/
	template <typename Dtype>
	void Base64DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int new_height = this->layer_param_.base64_data_param().new_height();
		const int new_width = this->layer_param_.base64_data_param().new_width();
		const bool is_color = this->layer_param_.base64_data_param().is_color();
		string datafile = this->layer_param_.base64_data_param().source();
		const int data_size = this->layer_param_.base64_data_param().data_size();
		 
		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";
		// Read the file with filenames and labels
		const string& source = this->layer_param_.base64_data_param().source();
		LOG(INFO) << "Opening Base64 file " << source;
		std::cout << source << std::endl;
		infile.open(source.c_str());
		string base64string;
		int label = 0;
		//string tmp;
		if (this->layer_param_.base64_data_param().shuffle()) {
			// randomly shuffle data
			// To be added
		}
		lines_id_ = 0;
		infile >> label >> base64string;
		//std::cout << "label: " << tmp << "image: " << base64string << std::endl;
		string decode_string = base64_decode(base64string);
		//std::cout << decode_string;
		vector<uchar> data(decode_string.begin(), decode_string.end());
		// Read an image, and use it to initialize the top blob.
		cv::Mat cv_img;
		if (is_color){
			cv_img = cv::imdecode(data, cv::IMREAD_COLOR);
		} else {
			cv_img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
		}
		std::cout << cv_img.cols << " " << cv_img.rows << " " << cv_img.channels() << std::endl;
		infile.seekg(0, ios::beg);
		//Debug
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(100);

		cv::imwrite("debug.jpeg", cv_img, compression_params);
		//CvMat cvMat = cv_img;
		//IplImage* image_jpg = cvDecodeImage(&cvMat);
		//cvSaveImage("debug.jpeg", image_jpg);

		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.base64_data_param().rand_skip()) {
			
		}

		CHECK(cv_img.data) << "Could not load line " << lines_id_;
		// Use data_transformer to infer the expected blob shape from a cv_image.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape prefetch_data and top[0] according to the batch_size.
		const int batch_size = this->layer_param_.base64_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";
		top_shape[0] = batch_size;
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape);
		}
		top[0]->Reshape(top_shape);

		LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
		// label
		vector<int> label_shape(1, batch_size);
		top[1]->Reshape(label_shape);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].label_.Reshape(label_shape);
		}
	}

	/*template <typename Dtype>
	void Base64DataLayer<Dtype>::ShuffleImages() {
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(lines_.begin(), lines_.end(), prefetch_rng);
	}*/

	// This function is called on prefetch thread
	template <typename Dtype>
	void Base64DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		double read_line_time = 0;
		double decode_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());
		CHECK(this->transformed_data_.count());
		Base64DataParameter base64_data_param = this->layer_param_.base64_data_param();
		const int batch_size = base64_data_param.batch_size(); 
		const int new_height = base64_data_param.new_height();
		const int new_width = base64_data_param.new_width();
		const bool is_color = base64_data_param.is_color();
		const int data_size = base64_data_param.data_size();
		string root_folder = base64_data_param.root_folder();

		string base64string;
		int label = 0;
		//string tmp;
		// Reshape according to the first image of each batch
		// on single input batches allows for inputs of varying dimension.
		infile >> label >> base64string;
		string decode_string = base64_decode(base64string);
		vector<uchar> data(decode_string.begin(), decode_string.end());
		// Read an image, and use it to initialize the top blob.
		cv::Mat cv_img;
		if (is_color){
			cv_img = cv::imdecode(data, cv::IMREAD_COLOR);
		} else {
			cv_img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
		}
		CHECK(cv_img.data) << "Could not load line " << lines_id_;
		// Use data_transformer to infer the expected blob shape from a cv_img.
		vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
		this->transformed_data_.Reshape(top_shape);
		// Reshape batch according to the batch_size.
		top_shape[0] = batch_size;
		batch->data_.Reshape(top_shape);

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();

		// datum scales
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			CHECK_GT(data_size, lines_id_);

			if (item_id != 0){
				infile >> label >> base64string;
				read_line_time += timer.MicroSeconds();
				timer.Start();
				string decode_string = base64_decode(base64string);
				decode_time += timer.MicroSeconds();
				timer.Start();
				//cout << decode_string;
				vector<uchar> data(decode_string.begin(), decode_string.end());
				// Read an image, and use it to initialize the top blob.
				if (is_color){
					cv_img = cv::imdecode(data, cv::IMREAD_COLOR);
				} else {
					cv_img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
				}
				CHECK(cv_img.data) << "Could not load line " << lines_id_;
			}
			read_time += timer.MicroSeconds();
			timer.Start();
			// Apply transformations (mirror, crop...) to the image
			int offset = batch->data_.offset(item_id);
			//cout << offset << endl;
			this->transformed_data_.set_cpu_data(prefetch_data + offset);
			this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
			trans_time += timer.MicroSeconds();

			prefetch_label[item_id] = label;
			// go to the next iter
			lines_id_++;
			if (lines_id_ >= data_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				lines_id_ = 0;
				infile.seekg(0, ios::beg);
			}
		}
		batch_timer.Stop();
		//cout << "Read line: " << read_line_time / 1000 << " ms.";
		//cout << "Basea64 decode: " << decode_time / 1000 << " ms.";
		//cout << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		//cout << "     Read time: " << read_time / 1000 << " ms.";
		//cout << "Transform time: " << trans_time / 1000 << " ms.";
		//DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		//DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		//DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(Base64DataLayer);
	REGISTER_LAYER_CLASS(Base64Data);

}  // namespace caffe
#endif  // USE_OPENCV
