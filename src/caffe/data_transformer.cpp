#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }

  // check if we want to use std_value
  has_std_values_ = false;
  if (param_.std_value_size() > 0) {
	  std_values_.clear();
	  for (int c = 0; c < param_.std_value_size(); ++c) {
		  std_values_.push_back(param_.std_value(c));
	  }
	  has_std_values_ = true;
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
	is_color_shift = false;
	if (datum.extra_type() == Datum_DataType_KL_INFO && datum.float_data_size() > 0 && param_.color_shift())
	{
		//for (int i = 0; i < 9; ++i)
		//{
		//	**(color_kl_cache_.P+i) = datum.float_data(i);
		//}
		memcpy(color_kl_cache_.P, datum.float_data().data(), sizeof(float)* 9);
		memcpy(color_kl_cache_.SqrtV, datum.float_data().data() + 9, sizeof(float)* 3);
		//for (int i = 0; i < 3; ++i)
		//{
		//	color_kl_cache_.SqrtV[i] = datum.float_data(i + 9);
		//}
		is_color_shift = true;
	}
	else if (param_.color_shift())
	{
		float P[] = { -0.5836, -0.6948, 0.4203, 
			-0.5808, -0.0045, -0.8140, 
			-0.5675, 0.7192, 0.4009 };
		float SqrtV[] = { 0.2175*255.0, 0.0188*255.0, 0.0045*255.0 };

		memcpy(color_kl_cache_.P, P, sizeof(float)* 9);
		memcpy(color_kl_cache_.SqrtV, SqrtV, sizeof(float)* 3);

		is_color_shift = true;
	}
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
	if (!cv_img.data)
	{
		transformed_blob->is_set_data(false);
		return;
	}
	else
		transformed_blob->is_set_data(true);
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }

	float min_scale = param_.multi_scale_param().min_scale();
	float max_scale = param_.multi_scale_param().max_scale();
	int min_length = param_.multi_scale_param().min_length();
	int max_length = param_.multi_scale_param().max_length();
	if (param_.has_multi_scale_param() &&
		param_.multi_scale_param().is_multi_scale() &&
		(max_scale >= min_scale && min_length<=max_length)){
#ifdef USE_OPENCV
		cv::Mat cv_img;
		DatumToCVMat(&datum, cv_img);
		if (!cv_img.data)
		{
			transformed_blob->is_set_data(false);
			return;
		}
		else
			transformed_blob->is_set_data(true);
		return Transform(cv_img, transformed_blob);
#else
		LOG(FATAL) << "resize image requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	}

  }

  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  transformed_blob->is_set_data(true);
  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
	cv::Mat tmp_cv_img;
	// random scale image
	static int flag_time = 0;
	static float cvt_time = 0;
	static float resize_time = 0;
	//clock_t st2 = clock();
	
	cv_img.convertTo(tmp_cv_img, CV_32F);
	//cvt_time += (clock()-st2);
	//if (flag_time % 2560 == 0)
	//{
	//	LOG(INFO) << "time convert image:" << (cvt_time) / CLOCKS_PER_SEC;
	//	cvt_time = 0;
	//}
	float min_scale = param_.multi_scale_param().min_scale();
	float max_scale = param_.multi_scale_param().max_scale();
	int min_length = param_.multi_scale_param().min_length();
	int max_length = param_.multi_scale_param().max_length();

	if (param_.has_multi_scale_param() &&
		param_.multi_scale_param().is_multi_scale() &&
		(max_scale >= min_scale && min_length<=max_length) ){
		int interpolation = cv::INTER_LINEAR; 
		switch (param_.interpolation())
		{
		case TransformationParameter_interpolation_type_INTER_LINEAR:
			interpolation = cv::INTER_LINEAR;
			break;
		case TransformationParameter_interpolation_type_INTER_CUBIC:
			interpolation = cv::INTER_CUBIC;
			break;
		case TransformationParameter_interpolation_type_INTER_LANCZOS4:
			interpolation = cv::INTER_LANCZOS4;
			break;
		default:
			break;
		}
		if (phase_ == TRAIN)
		{
			if (param_.multi_scale_param().padding())
			{
				int pad_width = param_.multi_scale_param().pad_width();
				cv::Mat  tmp = tmp_cv_img;
				//copyMakeBorder(tmp, tmp_cv_img, pad_width, pad_width,
				//	pad_width, pad_width, cv::BORDER_REPLICATE);
				cv::Scalar colorscalar;
				if (mean_values_.size() == 3)
				{
					colorscalar = cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]);
				}
				else if (mean_values_.size()==1)
				{
					colorscalar = cv::Scalar(mean_values_[0]);
				}
				else
				{
					colorscalar = cv::Scalar(0);
					LOG(WARNING) << "pad raw 0.";
				}

				copyMakeBorder(tmp, tmp_cv_img, pad_width, pad_width,
					pad_width, pad_width, cv::BORDER_CONSTANT,colorscalar);
				//copyMakeBorder(tmp, tmp_cv_img, pad_width, pad_width,
				//	pad_width, pad_width, cv::BORDER_REFLECT);
				int width = tmp_cv_img.rows;
				CHECK_EQ(width, max_length);
			}
			else
			{
				if (min_length == 0 || max_length == 0)
				{
					int org_height = cv_img.rows;
					int org_width = cv_img.cols;
					//get random scale
					int small_side = std::min(org_height, org_width);
					int min_side = small_side*min_scale;
					int max_side = small_side*max_scale;
					float scale = float(Rand(max_side - min_side + 1) + min_side) / float(small_side);

					if (param_.is_aspect_ration())
					{
						float aspect_ratio = float(Rand(8) + 9) / 12;
						int resize_height = org_height*scale;
						int resize_width = org_width*scale;
						if (Rand(2) == 0)
							resize_height = std::max((int)(org_height*scale*aspect_ratio), min_length);
						else
							resize_width = std::max((int)(org_width*scale*aspect_ratio), min_length);

						//cv::resize(cv_img, tmp_cv_img, cv::Size(resize_width, resize_height));
						cv::resize(tmp_cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
					}
					else
					{
						//scale image
						int resize_height = org_height*scale;
						int resize_width = org_width*scale;

						//cv::resize(cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
						cv::resize(tmp_cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
					}
				}
				else
				{
					int org_height = cv_img.rows;
					int org_width = cv_img.cols;
					//get random scale
					int small_side = std::min(org_height, org_width);
					//int min_side = small_side*min_scale;
					//int max_side = small_side*max_scale;
					float scale = float(Rand(max_length - min_length + 1) + min_length) / float(small_side);

					if (param_.is_aspect_ration())
					{
						float aspect_ratio = float(Rand(8) + 9) / 12;
						int resize_height = org_height*scale;
						int resize_width = org_width*scale;
						if (Rand(2) == 0)
							resize_height = std::max((int)(org_height*scale*aspect_ratio), min_length);
						else
							resize_width = std::max((int)(org_width*scale*aspect_ratio), min_length);

						// set to cropsize
						resize_height = std::max(resize_height, min_length);
						resize_width = std::max(resize_width, min_length);
						//cv::resize(cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
						cv::resize(tmp_cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
					}
					else
					{
						//scale image and set to cropsize
						int resize_height = std::max((int)(org_height*scale), min_length);
						int resize_width = std::max((int)(org_width*scale), min_length);

						//static int flag_time = 0;

						//clock_t st = clock();
						//cv::resize(cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
						cv::resize(tmp_cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
						//resize_time += (clock() - st);
						//if (flag_time % 2560 == 0)
						//{
						//	LOG(INFO) << "time resize image:" << (resize_time) / CLOCKS_PER_SEC;
						//	resize_time = 0;
						//}
						//flag_time++;
					}
				}
			}
		}
		else if (phase_ == TEST)
		{
			if (min_length == 0 || max_length == 0)
			{
				int org_height = cv_img.rows;
				int org_width = cv_img.cols;
				int resize_height = org_height*min_scale;
				int resize_width = org_width*min_scale;
				//cv::resize(cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
				cv::resize(tmp_cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
			}
			else
			{
				int org_height = cv_img.rows;
				int org_width = cv_img.cols;
				int small_side = std::min(org_height, org_width);
				float scale_ratio = float(min_length) / float(small_side);
				int resize_height = std::max((int)(org_height*scale_ratio),min_length);
				int resize_width = std::max((int)(org_width*scale_ratio),min_length);
				//std::cout << resize_height << " " << resize_width <<std::endl;
				//cv::resize(cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
				cv::resize(tmp_cv_img, tmp_cv_img, cv::Size(resize_width, resize_height), 0.0, 0.0, interpolation);
			}
		}
	}
	else
		tmp_cv_img = cv_img;

  const int crop_size = param_.crop_size();
  const int img_channels = tmp_cv_img.channels();
  const int img_height = tmp_cv_img.rows;
  const int img_width = tmp_cv_img.cols;

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  //CHECK(tmp_cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const Dtype scale = param_.scale();
  //const bool do_mirror = param_.mirror() && Rand(2);
  bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  bool use_crop_mean = false;
  if (has_mean_file) {
	  if (height == data_mean_.height() &&
		  width == data_mean_.width() &&
		  channels == data_mean_.channels())
	  {
		  use_crop_mean = true;
		  mean = data_mean_.mutable_cpu_data();
	  }
	  else
	  {
		  CHECK_EQ(img_channels, data_mean_.channels());
		  CHECK_EQ(img_height, data_mean_.height());
		  CHECK_EQ(img_width, data_mean_.width());
		  mean = data_mean_.mutable_cpu_data();
	  }
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  if (param_.test10crop())
  {
	  vector<int> h_off_set(10);
	  vector<int> w_off_set(10);
	  h_off_set = { 0, 0,
		  0,0, 
		  img_height - crop_size - 1, img_height - crop_size - 1, 
		  img_height - crop_size - 1, img_height - crop_size - 1
		  , (img_height - crop_size) / 2, (img_height - crop_size) / 2 };


	  w_off_set = { 0, 0, 
		  img_width - crop_size - 1, img_width - crop_size - 1,
		  0, 0, 
		  img_width - crop_size - 1, img_width - crop_size - 1,
		  (img_width - crop_size) / 2, (img_width - crop_size) / 2 };
	  for (int ncrop = 0; ncrop < 10; ncrop++)
	  {
		  int h_off = 0;
		  int w_off = 0;
		  do_mirror = ncrop % 2 == 0 ? false : true;
		  cv::Mat cv_cropped_img = tmp_cv_img;
		  if (crop_size) {
			  CHECK_EQ(crop_size, height);
			  CHECK_EQ(crop_size, width);
			  CHECK_EQ(phase_, TEST) << "test 10 crop only allowed on TEST phase.";
			  //h_off = (img_height - crop_size) / 2;
			  //w_off = (img_width - crop_size) / 2;
			  h_off = h_off_set[ncrop];
			  w_off = w_off_set[ncrop];
			  cv::Rect roi(w_off, h_off, crop_size, crop_size);
			  cv_cropped_img = tmp_cv_img(roi);
		  }
		  else {
			  CHECK_EQ(img_height, height);
			  CHECK_EQ(img_width, width);
		  }

		  CHECK(cv_cropped_img.data);

		  Dtype* transformed_data = transformed_blob->mutable_cpu_data()+transformed_blob->offset(ncrop);
		  int top_index;
		  for (int h = 0; h < height; ++h) {
			  //const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
			  const float* ptr = cv_cropped_img.ptr<float>(h);
			  int img_index = 0;
			  for (int w = 0; w < width; ++w) {
				  for (int c = 0; c < img_channels; ++c) {
					  if (do_mirror) {
						  top_index = (c * height + h) * width + (width - 1 - w);
					  }
					  else {
						  top_index = (c * height + h) * width + w;
					  }
					  // int top_index = (c * height + h) * width + w;
					  Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					  if (has_mean_file) {
						  int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
						  if (use_crop_mean)
						  {
							  transformed_data[top_index] =
								  (pixel - mean[top_index]) * scale;
						  }
						  else
						  {

							  transformed_data[top_index] =
								  (pixel - mean[mean_index]) * scale;
						  }
					  }
					  else {
						  if (has_mean_values) {
							  transformed_data[top_index] =
								  (pixel - mean_values_[c]) * scale;
						  }
						  else {
							  transformed_data[top_index] = pixel * scale;
						  }
					  }
				  }
			  }
		  }
	  }
  }
  else if (param_.has_rotate_num())
  {
	  int rotate_num = param_.rotate_num()+1;

	  int h_off = 0;
	  int w_off = 0;
	  cv::Mat cv_cropped_img = tmp_cv_img;

	  // calculate color shit vector.
	  Dtype color_shift[3] = { 0, 0, 0 };

	  for (int nrotate = 0; nrotate < rotate_num; nrotate++)
	  {
		  if (crop_size) {
			  CHECK_EQ(crop_size, height);
			  CHECK_EQ(crop_size, width);
			  // We only do random crop when we do training.
			  if (phase_ == TRAIN) {
				  h_off = Rand(img_height - crop_size + 1);
				  w_off = Rand(img_width - crop_size + 1);
			  }
			  else {
				  h_off = (img_height - crop_size) / 2;
				  w_off = (img_width - crop_size) / 2;
			  }

			  int min_rotate = param_.rotate_param().min_rotate();
			  int max_rotate = param_.rotate_param().max_rotate();

			  if (param_.has_rotate_param() &&
				  param_.rotate_param().is_rotate() &&
				  (min_rotate <= max_rotate)){
				  int rotate_angle = nrotate*(float(max_rotate - min_rotate) / (rotate_num-1)) + min_rotate;
				  //if (phase_ == TRAIN)
				  // rotate_angle = Rand(max_rotate - min_rotate + 1) + min_rotate;
				  //else
				  // rotate_angle = min_rotate;
				  cv::Point center = cv::Point((w_off + crop_size) / 2, (h_off + crop_size) / 2);
				  cv::Mat M = cv::getRotationMatrix2D(center, rotate_angle, 1);
				  cv::warpAffine(tmp_cv_img, tmp_cv_img, M, tmp_cv_img.size());
			  }
			  cv::Rect roi(w_off, h_off, crop_size, crop_size);
			  cv_cropped_img = tmp_cv_img(roi);
		  }
		  else {
			  CHECK_EQ(img_height, height);
			  CHECK_EQ(img_width, width);
			  int min_rotate = param_.rotate_param().min_rotate();
			  int max_rotate = param_.rotate_param().max_rotate();

			  if (param_.has_rotate_param() &&
				  param_.rotate_param().is_rotate() &&
				  (min_rotate <= max_rotate)){
				  int rotate_angle = nrotate*(float(max_rotate - min_rotate) / (rotate_num-1)) + min_rotate;
				  //int rotate_angle = Rand(max_rotate - min_rotate + 1) + min_rotate;
				  cv::Point center = cv::Point((img_width) / 2, (img_height) / 2);
				  cv::Mat M = cv::getRotationMatrix2D(center, rotate_angle, 1);
				  cv::warpAffine(tmp_cv_img, tmp_cv_img, M, tmp_cv_img.size());
			  }
		  }

		  CHECK(cv_cropped_img.data);

		  //Dtype* transformed_data = transformed_blob->mutable_cpu_data();
		  Dtype* transformed_data = transformed_blob->mutable_cpu_data() + transformed_blob->offset(nrotate);
		  int top_index;
		  for (int h = 0; h < height; ++h) {
			  //const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
			  const float* ptr = cv_cropped_img.ptr<float>(h);
			  int img_index = 0;
			  for (int w = 0; w < width; ++w) {
				  for (int c = 0; c < img_channels; ++c) {
					  if (do_mirror) {
						  top_index = (c * height + h) * width + (width - 1 - w);
					  }
					  else {
						  top_index = (c * height + h) * width + w;
					  }
					  // int top_index = (c * height + h) * width + w;
					  Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					  if (has_mean_file) {
						  int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
						  if (use_crop_mean)
						  {
							  transformed_data[top_index] =
								  (pixel - mean[top_index] + color_shift[c]) * scale;
						  }
						  else
						  {
							  transformed_data[top_index] =
								  (pixel - mean[mean_index] + color_shift[c]) * scale;
						  }
					  }
					  else {
						  if (has_mean_values) {
							  transformed_data[top_index] =
								  (pixel - mean_values_[c] + color_shift[c]) * scale;
						  }
						  else {
							  transformed_data[top_index] = (pixel + color_shift[c]) * scale;
						  }
					  }
				  }
			  }
		  }
	  }
  }
  else
  {
	  int h_off = 0;
	  int w_off = 0;
	  cv::Mat cv_cropped_img = tmp_cv_img;

	  // calculate color shit vector.
	  Dtype color_shift[3] = { 0, 0, 0 };
	  if (phase_ == TRAIN && is_color_shift)
	  {
		  // modify picture pixels respected to color KL matrix
		  Dtype a[3];
		  caffe_rng_gaussian<Dtype>(3, 0, 0.1, a);

		  for (int k = 0; k < 3; k++) 
		  {
			  for (int j = 0; j < 3; j++) 
			  {
				  color_shift[k] += color_kl_cache_.P[k][j] * color_kl_cache_.SqrtV[j] * a[j];
			  }
		  }
	//LOG(INFO) << color_shift[0] << "," << color_shift[1] << ","<< color_shift[2];
	  }
	  if (crop_size) {
		  CHECK_EQ(crop_size, height);
		  CHECK_EQ(crop_size, width);
		  // We only do random crop when we do training.
		  if (phase_ == TRAIN) {
			  h_off = Rand(img_height - crop_size + 1);
			  w_off = Rand(img_width - crop_size + 1);
		  }
		  else {
			  h_off = (img_height - crop_size) / 2;
			  w_off = (img_width - crop_size) / 2;
		  }

		  int min_rotate = param_.rotate_param().min_rotate();
		  int max_rotate = param_.rotate_param().max_rotate();

		  if (param_.has_rotate_param() &&
			  param_.rotate_param().is_rotate() &&
			  (min_rotate<=max_rotate)){
			  int rotate_angle = 0;
			  if (phase_==TRAIN || min_rotate<max_rotate)
				rotate_angle = Rand(max_rotate - min_rotate + 1) + min_rotate;
			  else
				rotate_angle =  min_rotate;
			  cv::Point center = cv::Point((w_off + crop_size) / 2, (h_off + crop_size) / 2);
			  cv::Mat M = cv::getRotationMatrix2D(center, rotate_angle,1);
			  cv::warpAffine(tmp_cv_img, tmp_cv_img, M,tmp_cv_img.size());
		  }
		  cv::Rect roi(w_off, h_off, crop_size, crop_size);
		  cv_cropped_img = tmp_cv_img(roi);
	  }
	  else {
		  CHECK_EQ(img_height, height);
		  CHECK_EQ(img_width, width);
		  int min_rotate = param_.rotate_param().min_rotate();
		  int max_rotate = param_.rotate_param().max_rotate();

		  if (param_.has_rotate_param() &&
			  param_.rotate_param().is_rotate() &&
			  (min_rotate <= max_rotate)){
			  int rotate_angle = Rand(max_rotate - min_rotate + 1) + min_rotate;
			  cv::Point center = cv::Point((img_width) / 2, (img_height) / 2);
			  cv::Mat M = cv::getRotationMatrix2D(center, rotate_angle,1);
			  cv::warpAffine(tmp_cv_img, tmp_cv_img, M, tmp_cv_img.size());
		  }
	  }

	  CHECK(cv_cropped_img.data);
	  cv::Mat gray_cropped_img;
	  bool is_color_jitter = false;
	  if (phase_ == TRAIN && param_.color_jitter())
	  {
		  
		  Dtype alpha;
		  //brightness
		  caffe_rng_uniform<Dtype>(1, -0.4, 0.4, &alpha);
		  alpha += 1; 
		  cv::Mat tmp_img = alpha*cv_cropped_img;
		  
		  // contrast
		  caffe_rng_uniform<Dtype>(1, -0.4, 0.4, &alpha);
		  alpha += 1;
		  cv::cvtColor(tmp_img, gray_cropped_img, cv::COLOR_BGR2GRAY);
		  tmp_img = alpha*tmp_img + 
			  (1 - alpha)*cv::mean(gray_cropped_img)[0];

		  //Saturation
		  caffe_rng_uniform<Dtype>(1, -0.4, 0.4, &alpha);
		  alpha += 1; 
		  cv::cvtColor(tmp_img, gray_cropped_img, cv::COLOR_BGR2GRAY);
		  cv::cvtColor(gray_cropped_img, gray_cropped_img, cv::COLOR_GRAY2BGR);
		  tmp_img = tmp_img*alpha + (1 - alpha)*gray_cropped_img;
		  
		  is_color_jitter = true;
		  cv_cropped_img = tmp_img;
	  }
	  cv::Mat cv_cropped_img_f32;
	  cv_cropped_img.convertTo(cv_cropped_img_f32, CV_32F);
	  cv_cropped_img = cv_cropped_img_f32;

	  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
	  int top_index;
	  for (int h = 0; h < height; ++h) {

		  //const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
		  const float* ptr = cv_cropped_img.ptr<float>(h);

		  int img_index = 0;
		  for (int w = 0; w < width; ++w) {
			  for (int c = 0; c < img_channels; ++c) {
				  if (do_mirror) {
					  top_index = (c * height + h) * width + (width - 1 - w);
				  }
				  else {
					  top_index = (c * height + h) * width + w;
				  }
				  Dtype tmp_scale = has_std_values_ ? 1.0 / std_values_[c] : 1;
				  // int top_index = (c * height + h) * width + w;
				  Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
				  if (has_mean_file) {
					  int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
					  if (use_crop_mean)
					  {
						  transformed_data[top_index] =
							  (pixel - mean[top_index] + color_shift[c]) * scale*tmp_scale;
					  }
					  else
					  {
						  transformed_data[top_index] =
							  (pixel - mean[mean_index] + color_shift[c]) * scale*tmp_scale;
					  }
				  }
				  else {
					  if (has_mean_values) {
						  transformed_data[top_index] =
							  (pixel - mean_values_[c] + color_shift[c]) * scale*tmp_scale;
					  }
					  else {
						  transformed_data[top_index] = (pixel + color_shift[c]) * scale*tmp_scale;
					  }
				  }
			  }
		  }
	  }
  }
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int crop_size = param_.crop_size();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop_size) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size, crop_size);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  const int crop_size = param_.crop_size();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  vector<int> shape(4);
  if (param_.test10crop())
  {
	  shape[0] = 10;
  }
  else{
	  shape[0] = 1;
  }
  shape[1] = datum_channels;
  shape[2] = (crop_size)? crop_size: datum_height;
  shape[3] = (crop_size)? crop_size: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  const int crop_size = param_.crop_size();
  const int img_channels = cv_img.channels();
   int img_height = cv_img.rows;
   int img_width = cv_img.cols;
  // Check dimensions.
  if (param_.multi_scale_param().is_multi_scale())
  {
	  int min_length = param_.multi_scale_param().min_length();
	  int max_length = param_.multi_scale_param().max_length();
	  if (min_length > 0 && max_length > 0)
	  {
		  CHECK_GE(max_length, min_length);
		  int min_side = std::min(img_height, img_width);
		  float scale = float(min_length)/float(min_side);
		  img_height = std::max((int)(img_height*scale),min_length);
		  img_width = std::max((int)(img_width* scale),min_length);
	  }
	  else
	  {
		  float min_scale = param_.multi_scale_param().min_scale();
		  float max_scale = param_.multi_scale_param().max_scale();
		  CHECK_GE(max_scale, min_scale);
		  img_height = img_height*min_scale;
		  img_width = img_width*min_scale;
	  }
  }

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  // Build BlobShape.
  vector<int> shape(4);
  if (param_.test10crop())
  {
	  shape[0] = 10;
  }
  else
  {
	  shape[0] = 1;
  }
  shape[1] = img_channels;
  shape[2] = (crop_size)? crop_size: img_height;
  shape[3] = (crop_size)? crop_size: img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size()) || 
	  (phase_ == TRAIN && param_.multi_scale_param().is_multi_scale()
	  || (phase_ == TEST && param_.rotate_param().is_rotate()));
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
