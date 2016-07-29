#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	vector<int> rand_sequence(vector<float>& ratio_vec)
	{
		vector<int> result_sequence;
		vector<int> coef_set(ratio_vec.size());
		float total_sum = 0;
		int num_rand = 10000;
		for (int i = 0; i < coef_set.size(); ++i)
		{
			total_sum += ratio_vec[i];
		}
		for (int i = 0; i < coef_set.size(); ++i)
		{
			coef_set[i] = (int)(ratio_vec[i] / total_sum * num_rand);
		}
		for (int i = 1; i < coef_set.size(); ++i)
		{
			coef_set[i] += coef_set[i - 1];
		}
		int idx = coef_set.size() - 1;
		while (idx>=0 && ratio_vec[idx] == 0)
		{
			idx--;
		}
		if (idx>=0)
			coef_set[idx] = num_rand;
		result_sequence.resize(num_rand);
		int cur_idx = 0;
		for (int i = 0; i < result_sequence.size(); ++i)
		{
			while (i >= coef_set[cur_idx])
			{
				cur_idx++;
				CHECK(cur_idx < coef_set.size()) << "exceed cursor set size";
			}
			result_sequence[i] = cur_idx;
		}
		std::srand(std::time(0));
		std::random_shuffle(result_sequence.begin(), result_sequence.end());

		//std::ofstream out_ran("tmp2.txt");
		//for (int ri = 0; ri<result_sequence.size(); ++ri)
		//{
		//	out_ran << result_sequence[ri] << std::endl;
		//}
		return result_sequence;
	}

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
		use_key_files = false;
		shuffle = false;

		if (param_.data_param().key_files_size() > 0 || param_.data_param().shuffle())
		{
			if (param_.data_param().key_files_size() > 0)
			{
				use_key_files = true;
				cursor->set_key_by_file(param_.data_param().key_files(0));
			}
			if (param_.data_param().shuffle())
			{
				shuffle = true;
				cursor->shuffle();
			}
			cursor->SeekToFirstKey();
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
		db_set.clear();
		cursor_set.clear();
		for (int i = 0; i < param_.data_param().multi_source_size(); ++i)
		{
			if (param_.data_param().multi_backends_size()>0)
			{
				shared_ptr<db::DB> db(db::GetDB(param_.data_param().multi_backends(i)));
				db->Open(param_.data_param().multi_source(i), db::READ);
				db_set.push_back(db);

				shared_ptr<db::Cursor> cursor(db->NewCursor());
				cursor_set.push_back(cursor);
			}
			else
			{
				shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
				db->Open(param_.data_param().multi_source(i), db::READ);
				db_set.push_back(db);

				shared_ptr<db::Cursor> cursor(db->NewCursor());
				cursor_set.push_back(cursor);
			}
		}
		
		//random sequence.
		vector<float> ratio_vec;
		for (int i = 0; i<param_.data_param().ratio_sample_size(); ++i)
		{
			ratio_vec.push_back(param_.data_param().ratio_sample(i));
		}
		random_sequence = rand_sequence(ratio_vec);

		vector<float> sum_ratio(ratio_vec.size(), 0);
		for (int ri = 0; ri < random_sequence.size(); ++ri)
		{
			sum_ratio[random_sequence[ri]] += 1;
		}
		string debug_ratio_str = "sample ratio: ";
		for (int ri = 0; ri < sum_ratio.size(); ++ri)
			debug_ratio_str = debug_ratio_str + std::to_string(sum_ratio[ri])+",";
		LOG(INFO) << debug_ratio_str;
		//std::ofstream out_ran("tmp.txt");
		//for (int ri = 0; ri<random_sequence.size(); ++ri)
		//{
		//	out_ran << random_sequence[ri] << "," ;
		//}
		// read key file and shuffle
		use_key_files = false;
		shuffle = false;
		if (param_.data_param().key_files_size() > 0 || param_.data_param().shuffle())
		{
			for (int idx = 0; idx < cursor_set.size(); ++idx)
			{
				if (param_.data_param().key_files_size() > idx)		// has key list file and read them
				{
					use_key_files = true;
					cursor_set[idx]->set_key_by_file(param_.data_param().key_files(idx));
				}
				if (param_.data_param().shuffle())
				{
					shuffle = true;
					cursor_set[idx]->shuffle();
				}
				cursor_set[idx]->SeekToFirstKey();
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
					if (need_reset_list_)
					{
						//CHECK(name_list_.size(), cursor_set.size()) << "name size not equal to cursor set size.";
						//CHECK(ratio_list_.size(), cursor_set.size()) << "ratio size not equal to cursor set size.";
						for (int idx = 0; idx < cursor_set.size(); ++idx)
						{
							cursor_set[idx]->set_key_by_file(name_list_[idx]);
							if (ratio_list_[idx]>0)
							{
								cursor_set[idx]->shuffle();
								cursor_set[idx]->SeekToFirstKey();
							}
						}
						random_sequence = rand_sequence(ratio_list_);
						cursor_idx = 0;
						LOG(INFO) << "reset data base key list: ";
						for (int idx = 0; idx < name_list_.size(); ++idx)
						{
							LOG(INFO) << name_list_[idx] << " : " << ratio_list_[idx];
						}
						need_reset_list_ = false;
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

		if (cursor->get_use_other_label())
			datum->set_label(cursor->label());

		//int pos = cursor->key().find_first_of('/');
		//string image_name = cursor->key().substr(pos + 1);
		//string out_name = "tmp/" + std::to_string(random_sequence[cursor_idx]) + "_" + image_name;
		//LOG(INFO) << out_name;
		//std::ofstream our_img(out_name,std::ofstream::binary);
		//our_img << datum->data();
		//our_img.close();

		if (cursor->get_use_kl_info())
		{
			datum->mutable_float_data()->Resize(cursor->kl_info().size(), 0);

			memcpy(datum->mutable_float_data()->mutable_data(), cursor->kl_info().data(),
				sizeof(float)*cursor->kl_info().size());
		}

		qp->full_.push(datum);
		if (!cursor->valid())
		{
			LOG(INFO) << "invalid key";
		}
		// go to the next iter
		if (shuffle || use_key_files)
		{
			cursor->NextKey();
			if (!cursor->valid())
			{
				LOG(INFO) << "Restarting data and shuffle.";
				if (shuffle)
					cursor->shuffle();
				cursor->SeekToFirstKey();
			}
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

		if (cursor->get_use_other_label())
		{
			datum->set_label(cursor->label());
		}
		
		//int pos = cursor->key().find_first_of('/');
		//string image_name = cursor->key().substr(pos + 1);
		//string out_name = "tmp/" + std::to_string(random_sequence[cursor_idx]) + "_" + image_name;
		//LOG(INFO) << out_name;
		//std::ofstream our_img(out_name, std::ofstream::binary);
		//our_img << datum->data();
		//our_img.close();

		if (cursor->get_use_kl_info())
		{
			datum->mutable_float_data()->Resize(cursor->kl_info().size(), 0);
			memcpy(datum->mutable_float_data()->mutable_data(),
				cursor->kl_info().data(), sizeof(float)*cursor->kl_info().size());
		}
		qp->full_.push(datum);

		if (!cursor->valid())
		{
			LOG(INFO) << "invalid key";
		}
		// go to the next iter
		if (shuffle||use_key_files)
		{
			cursor->NextKey();
			if (!cursor->valid())
			{
				LOG(INFO) << "Restarting data and shuffle.";
				if (shuffle)
					cursor->shuffle();
				cursor->SeekToFirstKey();
			}
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
