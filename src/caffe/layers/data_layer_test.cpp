#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer_test.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataTestLayer<Dtype>::DataTestLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataTestLayer<Dtype>::~DataTestLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataTestLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());
  // Read if test on 10 crop view.
  test10crop_ = this->transform_param_.test10crop();
  test_rotate_ = this->transform_param_.has_rotate_num();
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  if (test10crop_)
	  top_shape[0] = batch_size;
  else if (test_rotate_)
	  top_shape[0] = this->transform_param_.rotate_num() + 1;
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (test10crop_)
  {
	  num_per_img_ = 10;
	  CHECK(batch_size % num_per_img_ == 0) << "batch size must be divided by 10 for test on 10 crop.";
	  if (this->output_labels_) {
		  vector<int> label_shape(1, batch_size / num_per_img_);
		  top[1]->Reshape(label_shape);
		  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			  this->prefetch_[i].label_.Reshape(label_shape);
		  }
	  }
  }
  else if (test_rotate_)
  {
	  num_per_img_ = this->transform_param_.rotate_num() + 1;
	  CHECK(batch_size % num_per_img_ == 0) << "batch size must be divided by 10 for test on 10 crop.";
	  if (this->output_labels_) {
		  vector<int> label_shape(1, batch_size / num_per_img_);
		  top[1]->Reshape(label_shape);
		  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			  this->prefetch_[i].label_.Reshape(label_shape);
		  }
	  }
  }
  else
  {
	  if (this->output_labels_) {
		  vector<int> label_shape(1, batch_size);
		  top[1]->Reshape(label_shape);
		  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			  this->prefetch_[i].label_.Reshape(label_shape);
		  }
	  }
  }

}

// This function is called on prefetch thread
template<typename Dtype>
void DataTestLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  if (test10crop_ || test_rotate_)
	  top_shape[0] = num_per_img_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  if (test10crop_ || test_rotate_)
  {
	  for (int item_id = 0; item_id < batch_size / num_per_img_; ++item_id) {
		  timer.Start();
		  // get a datum
		  Datum& datum = *(reader_.full().pop("Waiting for data"));
		  read_time += timer.MicroSeconds();
		  timer.Start();
		  // Apply data transformations (mirror, scale, crop...)
		  int offset = batch->data_.offset(item_id*num_per_img_);
		  this->transformed_data_.set_cpu_data(top_data + offset);
		  this->data_transformer_->Transform(datum, &(this->transformed_data_));
		  // Copy label.
		  if (this->output_labels_) {
			  top_label[item_id] = datum.label();
		  }
		  trans_time += timer.MicroSeconds();

		  reader_.free().push(const_cast<Datum*>(&datum));
	  }
  }
  else
  {
	  for (int item_id = 0; item_id < batch_size; ++item_id) {
		  timer.Start();
		  // get a datum
		  Datum& datum = *(reader_.full().pop("Waiting for data"));
		  read_time += timer.MicroSeconds();
		  timer.Start();
		  // Apply data transformations (mirror, scale, crop...)
		  int offset = batch->data_.offset(item_id);
		  this->transformed_data_.set_cpu_data(top_data + offset);
		  this->data_transformer_->Transform(datum, &(this->transformed_data_));
		  // Copy label.
		  if (this->output_labels_) {
			  top_label[item_id] = datum.label();
		  }
		  trans_time += timer.MicroSeconds();

		  reader_.free().push(const_cast<Datum*>(&datum));
	  }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataTestLayer);
REGISTER_LAYER_CLASS(DataTest);

}  // namespace caffe
