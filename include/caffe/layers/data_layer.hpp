#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {
	template<typename Dtype>
	class TransformThread : public DataTransformer<Dtype>, public InternalThread
	{
	public:
		TransformThread(const TransformationParameter& param, Phase phase):
			DataTransformer<Dtype>(param,phase)
		{
		}
		void init(
			BlockingQueue<std::pair<Datum*, Blob<Dtype>*>>* transform_full,
			BlockingQueue<Blob<Dtype>*>* transform_free,
			BlockingQueue<Datum*>* datum_free);

		~TransformThread(){
			StopInternalThread();
		}
	protected:
		virtual void InternalThreadEntry();

		BlockingQueue<std::pair<Datum*, Blob<Dtype>*>>* transform_data_full_;
		BlockingQueue<Blob<Dtype>*>* transform_data_free_;
		BlockingQueue<Datum*>* datum_free_;
	};

	template<typename Dtype>
	class TransformParallel{
	public:
		explicit TransformParallel(const TransformationParameter& param, Phase phase, int num_cache = 100, int num_trans = 3);

		void Reshape(vector<int> shape);

		void TransformOne(Datum* datum, Dtype* transformed_data);
		bool CheckCompleted();
		~TransformParallel();
		void return_datum(BlockingQueue<Datum*>* datum_queue)
		{
			Datum* tmp_datum;
			while (datum_free_.try_pop(&tmp_datum))
			{
				datum_queue->push(tmp_datum);
			}
		}
	protected:
		int num_transformer_;
		int cache_size_;
		vector<TransformThread<Dtype>*> transformers_set_;
		BlockingQueue<std::pair<Datum*, Blob<Dtype>*>> transform_data_full_;
		BlockingQueue<Blob<Dtype>*> transform_data_free_;
		BlockingQueue<Datum*> datum_free_;
	};


template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  void reset_list(vector<string>& name_list, vector<float>& ratio_list)
  {
	  reader_.reset_list(name_list, ratio_list);
  }
 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;

  shared_ptr<TransformParallel<Dtype>> transform_par_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
