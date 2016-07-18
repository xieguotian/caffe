#ifndef CAFFE_DATA_READER_HPP_
#define CAFFE_DATA_READER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"

namespace caffe {

/**
 * @brief Reads data from a source to queues available to data layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */
class DataReader {
 public:
  explicit DataReader(const LayerParameter& param);
  ~DataReader();

  inline BlockingQueue<Datum*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Datum*>& full() const {
    return queue_pair_->full_;
  }

  void reset_list(vector<string>& name_list, vector<float>& ratio_list)
  {
	  body_->reset_list(name_list, ratio_list);
  }
 protected:
  // Queue pairs are shared between a body and its readers
  class QueuePair {
   public:
    explicit QueuePair(int size);
    ~QueuePair();

    BlockingQueue<Datum*> free_;
    BlockingQueue<Datum*> full_;

  DISABLE_COPY_AND_ASSIGN(QueuePair);
  };

  // A single body is created per source
  class Body : public InternalThread {
   public:
    explicit Body(const LayerParameter& param);
    virtual ~Body();
	void reset_list(vector<string>& name_list, vector<float>& ratio_list)
	{
		need_reset_list_ = true;
		name_list_ =  name_list;
		ratio_list_ = ratio_list;
	}
   protected:
    void InternalThreadEntry();
    void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    BlockingQueue<shared_ptr<QueuePair> > new_queue_pairs_;

    friend class DataReader;

	// marker reset_list
	bool need_reset_list_ = false;
	vector<string> name_list_;
	vector<float> ratio_list_;

	// shuffle each epoch
	bool shuffle;
	vector<string> key_list;
	vector<int> label_list;
	vector<int> key_index;
	vector<vector<float>> kl_info_vec;

	int key_position;
	bool use_other_label = false;
	bool use_key_files = false;

	vector<vector<string>> key_list_set;
	vector<vector<int>> label_list_set;
	vector<vector<int>> key_index_set;
	vector<vector<vector<float>>> kl_info_set;
	vector<int> key_pos_set;
	vector<int> random_sequence;
	int cursor_idx;
  DISABLE_COPY_AND_ASSIGN(Body);
  };

  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_param().source();
  }

  const shared_ptr<QueuePair> queue_pair_;
  shared_ptr<Body> body_;

  static map<const string, boost::weak_ptr<DataReader::Body> > bodies_;

DISABLE_COPY_AND_ASSIGN(DataReader);
};

}  // namespace caffe

#endif  // CAFFE_DATA_READER_HPP_
