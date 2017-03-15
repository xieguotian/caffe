#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  //this->transform_par_.reset(new TransformParallel<Dtype>(transform_param_, this->phase_));
  //this->transform_par_->Reshape(top_shape);

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
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

  //extra_data
  this->output_extra_data_ = false;
  if (top.size()>2)
  {
	  this->output_extra_data_ = true;
	  vector<int> extra_shape(2);// (batch_size, datum.float_data_size());
	  extra_shape[0] = batch_size;
	  extra_shape[1] = datum.float_data_size();
	  top[2]->Reshape(extra_shape);
	  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		  this->prefetch_[i].extra_data_.Reshape(extra_shape);
	  }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
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
  this->transformed_data_.Reshape(top_shape);
  //this->transform_par_->Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
	this->transformed_data_.is_set_data(false);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

	while (!this->transformed_data_.has_set_data())
	{
		LOG(INFO) << "transform data fail";
		reader_.free().push(const_cast<Datum*>(&datum));
		datum = *(reader_.full().pop("Waiting for data"));
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
	}
	//this->transform_par_->TransformOne(&datum, top_data + offset);
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
	//copy extra_data.
	if (this->output_extra_data_)
	{
		Dtype* extra_ptr = batch->extra_data_.mutable_cpu_data() + batch->extra_data_.offset(item_id);
		for (int extra_id = 0; extra_id < datum.float_data_size(); extra_id++)
			extra_ptr[extra_id] = datum.float_data(extra_id);
	}
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  //this->transform_par_->CheckCompleted();
  //this->transform_par_->return_datum(&reader_.free());
  timer.Stop();
  batch_timer.Stop();
  
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
TransformParallel<Dtype>::TransformParallel(const TransformationParameter& param, Phase phase, int num_cache, int num_trans){
	cache_size_ = num_cache;
	for (int i = 0; i < cache_size_; i++)
	{
		transform_data_free_.push(new Blob<Dtype>());
	}

	num_transformer_ = num_trans;
	transformers_set_.resize(num_transformer_);
	for (int i = 0; i < num_transformer_; i++)
	{
		transformers_set_[i] = new TransformThread<Dtype>(param, phase);
		transformers_set_[i]->init(&transform_data_full_, &transform_data_free_,&datum_free_);
	}

}

template<typename Dtype>
void TransformParallel<Dtype>::Reshape(vector<int> shape)
{
	for (int i = 0; i < cache_size_; i++)
	{
		Blob<Dtype>* cache_tmp = transform_data_free_.pop();
		cache_tmp->Reshape(shape);
		transform_data_free_.push(cache_tmp);
	}
}

template<typename Dtype>
void TransformParallel<Dtype>::TransformOne(Datum* datum, Dtype* transformed_data)
{
	Blob<Dtype>* transform_data = transform_data_free_.pop();
	transform_data->set_cpu_data(transformed_data);
	std::pair<Datum*, Blob<Dtype>*> transform_data_pair(datum, transform_data);
	transform_data_full_.push(transform_data_pair);
}

template<typename Dtype>
bool TransformParallel<Dtype>::CheckCompleted(){
	while (!(transform_data_free_.size() == cache_size_))
	{
	}
	return true;
}

template<typename Dtype>
TransformParallel<Dtype>::~TransformParallel()
{
	for (int i = 0; i < transformers_set_.size(); i++)
	{
		transformers_set_[i]->StopInternalThread();
		delete transformers_set_[i];
	}
	std::pair<Datum*, Blob<Dtype>*> tmp_pair;
	while (transform_data_full_.try_pop(&tmp_pair))
	{
		delete tmp_pair.second;
	}

	Blob<Dtype>* tmp_data;
	while (transform_data_free_.try_pop(&tmp_data))
	{
		delete tmp_data;
	}
}

template<typename Dtype>
void TransformThread<Dtype>::InternalThreadEntry()
{
	while (true)
	{
		std::pair<Datum*, Blob<Dtype>*> data_pair = transform_data_full_->pop();
		Datum* data_tmp = data_pair.first;
		Blob<Dtype>* transform_data = data_pair.second;
		this->Transform(*data_tmp, transform_data);
		transform_data_free_->push(const_cast<Blob<Dtype>*>(transform_data));
		datum_free_->push(const_cast<Datum*>(data_tmp));
	}
}

template<typename Dtype>
void TransformThread<Dtype>::init(
	BlockingQueue<std::pair<Datum*, Blob<Dtype>*>>* transform_full,
	BlockingQueue<Blob<Dtype>*>* transform_free,
	BlockingQueue<Datum*>* datum_free)
{
	transform_data_full_ = transform_full;
	transform_data_free_ = transform_free;
	datum_free_ = datum_free;
	InitRand();
	StartInternalThread();
}
INSTANTIATE_CLASS(TransformThread);
INSTANTIATE_CLASS(TransformParallel);

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
