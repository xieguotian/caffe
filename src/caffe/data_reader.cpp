#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using boost::weak_ptr;

map<const string, weak_ptr<DataReader::Body> > DataReader::bodies_;
static boost::mutex bodies_mutex_;

DataReader::DataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.data_param().prefetch() * param.data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
  body_->shuffle = param.data_param().shuffle();
}

DataReader::~DataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

//

DataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of datums
  for (int i = 0; i < size; ++i) {
    free_.push(new Datum());
  }
}

DataReader::QueuePair::~QueuePair() {
  Datum* datum;
  while (free_.try_pop(&datum)) {
    delete datum;
  }
  while (full_.try_pop(&datum)) {
    delete datum;
  }
}

//

DataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

DataReader::Body::~Body() {
  StopInternalThread();
}

void DataReader::Body::InternalThreadEntry() {
	if (param_.data_param().has_source())
	{
		shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
		db->Open(param_.data_param().source(), db::READ);
		shared_ptr<db::Cursor> cursor(db->NewCursor());
		if (shuffle)
		{
			key_list.clear();
			label_list.clear();
			if (param_.data_param().key_files_size() > 0)
			{
				std::ifstream key_file(param_.data_param().key_files(0));
				string key;
				int label;
				use_other_label = true;
				while (key_file >> key >>label)
				{
					key_list.push_back(key);
					label_list.push_back(label);
				}
			}
			else
			{
				use_other_label = false;
				while (cursor->valid())
				{
					string key = cursor->key();
					key_list.push_back(key);
					cursor->Next();
				}

				cursor->SeekToFirst();
			}
			std::srand(std::time(0));
			for (int i = 0; i < key_list.size(); ++i)
			{
				key_index.push_back(i);
			}
			std::random_shuffle(key_index.begin(), key_index.end());
			key_position = 0;
			cursor->SeekByKey(key_list[key_index[key_position]]);
		}
		vector<shared_ptr<QueuePair> > qps;
		try {
			int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

			// To ensure deterministic runs, only start running once all solvers
			// are ready. But solvers need to peek on one item during initialization,
			// so read one item, then wait for the next solver.
			for (int i = 0; i < solver_count; ++i) {
				shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
				read_one(cursor.get(), qp.get());
				qps.push_back(qp);
			}
			// Main loop
			while (!must_stop()) {
				for (int i = 0; i < solver_count; ++i) {
					read_one(cursor.get(), qps[i].get());
				}
				// Check no additional readers have been created. This can happen if
				// more than one net is trained at a time per process, whether single
				// or multi solver. It might also happen if two data layers have same
				// name and same source.
				CHECK_EQ(new_queue_pairs_.size(), 0);
			}
		}
		catch (boost::thread_interrupted&) {
			// Interrupted exception is expected on shutdown
		}
	}
	else // read multi-source db with specific sample ratio
	{
		//read all data base.
		vector<shared_ptr<db::DB>> db_set;
		vector < shared_ptr<db::Cursor>> cursor_set;
		for (int i = 0; i < param_.data_param().multi_source_size(); ++i)
		{
			shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
			db->Open(param_.data_param().multi_source(i), db::READ);
			db_set.push_back(db);

			shared_ptr<db::Cursor> cursor(db->NewCursor());
			cursor_set.push_back(cursor);
		}
		
		//random sequence.
		vector<int> coef_set(param_.data_param().ratio_sample_size());
		float total_sum = 0;
		for (int i = 0; i < coef_set.size(); ++i)
		{
			total_sum += param_.data_param().ratio_sample(i);
		}
		for (int i = 0; i < coef_set.size(); ++i)
		{
			coef_set[i] = (int)(param_.data_param().ratio_sample(i) / total_sum * 1000);
		}
		for (int i = 1; i < coef_set.size(); ++i)
		{
			coef_set[i] += coef_set[i - 1];
		}
		random_sequence.resize(1000);
		int cur_idx = 0;
		for (int i = 0; i < random_sequence.size();++i)
		{
			if (i>=coef_set[cur_idx])
				cur_idx++;
			random_sequence[i] = cur_idx;
		}
		std::random_shuffle(random_sequence.begin(), random_sequence.end());

		//shuffle all data base
		if (shuffle)
		{
			key_index_set.clear();
			key_list_set.clear();
			key_pos_set.clear();
			label_list_set.clear();

			key_index_set.resize(cursor_set.size());
			key_list_set.resize(cursor_set.size());
			key_pos_set.resize(cursor_set.size());
			label_list_set.resize(cursor_set.size());

			for (int idx = 0; idx < cursor_set.size(); ++idx)
			{
				shared_ptr<db::Cursor> cursor = cursor_set[idx];

				key_list_set[idx].clear();
				label_list_set[idx].clear();
				if (param_.data_param().key_files_size() > idx)		// has key list file and read them
				{
					std::ifstream key_file(param_.data_param().key_files(idx));
					string key;
					int label;
					use_other_label = true;
					while (key_file >> key>>label)
					{
						key_list_set[idx].push_back(key);
						label_list_set[idx].push_back(label);
					}
				}
				else           // do not has key list file and read from data base.
				{
					use_other_label = false;
					while (cursor->valid())
					{
						string key = cursor->key();
						//key_list.push_back(key);
						key_list_set[idx].push_back(key);
						cursor->Next();
					}
					cursor->SeekToFirst();
				}

				// random shuffle key.
				std::srand(std::time(0));
				for (int i = 0; i < key_list_set[idx].size(); ++i)
				{
					key_index_set[idx].push_back(i);
				}
				std::random_shuffle(key_index_set[idx].begin(), key_index_set[idx].end());

				// read first key and data.
				key_pos_set[idx] = 0;
				cursor->SeekByKey(key_list_set[idx][key_index_set[idx][key_pos_set[idx]]]);
			}
		}
		vector<shared_ptr<QueuePair> > qps;
		try {
			cursor_idx = 0;
			int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

			// To ensure deterministic runs, only start running once all solvers
			// are ready. But solvers need to peek on one item during initialization,
			// so read one item, then wait for the next solver.
			for (int i = 0; i < solver_count; ++i) {
				shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
				read_one(cursor_set[random_sequence[cursor_idx]].get(), qp.get());
				qps.push_back(qp);
				cursor_idx++;
				if (cursor_idx >= random_sequence.size())
				{
					cursor_idx = 0;
					std::random_shuffle(random_sequence.begin(), random_sequence.end());
				}
			}
			// Main loop
			while (!must_stop()) {
				for (int i = 0; i < solver_count; ++i) {
					read_one(cursor_set[random_sequence[cursor_idx]].get(), qps[i].get());
					cursor_idx++;
					if (cursor_idx >= random_sequence.size())
					{
						cursor_idx = 0;
						std::random_shuffle(random_sequence.begin(), random_sequence.end());
					}
				}
				// Check no additional readers have been created. This can happen if
				// more than one net is trained at a time per process, whether single
				// or multi solver. It might also happen if two data layers have same
				// name and same source.
				CHECK_EQ(new_queue_pairs_.size(), 0);
			}
		}
		catch (boost::thread_interrupted&) {
			// Interrupted exception is expected on shutdown
		}
	}
}

void DataReader::Body::read_one(db::Cursor* cursor, QueuePair* qp) {
	if (param_.data_param().has_source())
	{

		Datum* datum = qp->free_.pop();
		// TODO deserialize in-place instead of copy?
		datum->ParseFromString(cursor->value());
		if (use_other_label)
			datum->set_label(label_list[key_index[key_position]]);

		qp->full_.push(datum);

		DLOG(INFO) << key_list[key_index[key_position]] << " vs " << cursor->key() << " label: " << datum->label();
		if (!cursor->valid())
			LOG(INFO) << "invalid key: " << key_list[key_index[key_position]];

		// go to the next iter
		if (shuffle)
		{
			key_position++;
			if (key_position >= key_index.size())
			{
				LOG(INFO) << "Restarting data and shuffle.";
				std::random_shuffle(key_index.begin(), key_index.end());
				key_position = 0;
			}
			cursor->SeekByKey(key_list[key_index[key_position]]);
		}
		else
		{
			cursor->Next();
			if (!cursor->valid()) {
				DLOG(INFO) << "Restarting data prefetching from start.";
				cursor->SeekToFirst();
			}
		}
	}
	else
	{
		int cur_idx = random_sequence[cursor_idx];
		Datum* datum = qp->free_.pop();
		// TODO deserialize in-place instead of copy?
		datum->ParseFromString(cursor->value());
		if (use_other_label)
		{
			datum->set_label(label_list_set[cur_idx][key_index_set[cur_idx][key_pos_set[cur_idx]]]);
			//LOG(INFO) << "orgin label vs new label: " << datum->label() << " vs " << label_list_set[cur_idx][key_index_set[cur_idx][key_pos_set[cur_idx]]];
			//LOG(INFO) << "cur_idx: " << cur_idx << ",key_pos: " << key_pos_set[cur_idx] << ",key_indes: " << key_index_set[cur_idx][key_pos_set[cur_idx]]
			//	<< ",key: " << key_list_set[cur_idx][key_index_set[cur_idx][key_pos_set[cur_idx]]];
		}

		qp->full_.push(datum);
		DLOG(INFO) << key_list_set[cur_idx][key_index_set[cur_idx][key_pos_set[cur_idx]]] << " vs " << cursor->key() << " label: " << datum->label();
		if (!cursor->valid())
			LOG(INFO) << "invalid key: " << key_list_set[cur_idx][key_index_set[cur_idx][key_pos_set[cur_idx]]];
		// go to the next iter
		if (shuffle)
		{
			key_pos_set[cur_idx]++;
			if (key_pos_set[cur_idx] >= key_index_set[cur_idx].size())
			{
				LOG(INFO) << "Restarting data and shuffle.";
				std::random_shuffle(key_index_set[cur_idx].begin(), key_index_set[cur_idx].end());
				key_pos_set[cur_idx] = 0;
			}
			cursor->SeekByKey(key_list_set[cur_idx][key_index_set[cur_idx][key_pos_set[cur_idx]]]);
		}
		else
		{
			cursor->Next();
			if (!cursor->valid()) {
				DLOG(INFO) << "Restarting data prefetching from start.";
				cursor->SeekToFirst();
			}
		}
	}
}

}  // namespace caffe
