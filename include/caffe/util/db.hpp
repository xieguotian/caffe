#ifndef CAFFE_UTIL_DB_HPP
#define CAFFE_UTIL_DB_HPP

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe { namespace db {

enum Mode { READ, WRITE, NEW };

class Cursor {
 public:
  Cursor() {
	  key_index_list.clear();
	  key_list.clear();
	  label_list.clear();
	  kl_info_vec.clear();
	  valid_ = false;
	  is_shuffle = false;
  }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual bool valid() = 0;

  virtual void SeekByKey(string key) = 0;
  virtual void set_key_by_file(string key_file);
  virtual void shuffle();
  virtual void NextKey(){
	  if (key_pos >= key_index_list.size())
	  {
		  if (is_shuffle)
		  {
			  LOG(INFO) << "shuffle and restarting data.";
			  shuffle();
			  SeekToFirstKey();
		  }
		  else
			valid_ = false;
	  }
	  else
	  {
		  SeekByKey(key_list[key_index_list[key_pos]]);
		  key_pos++;
		  while (!valid_ && key_pos < key_index_list.size())
		  {
			  LOG(INFO) << "Read current data fail and read next.";
			  SeekByKey(key_list[key_index_list[key_pos]]);
			  key_pos++;
		  }
	  }
  }
  virtual void SeekToFirstKey() {
	  if (key_index_list.size() == 0)
		  get_key_from_db();

	  key_pos = 0;
	  NextKey();
  }
  virtual bool get_use_other_label()
  {
	  return label_list.size() > 0;
  }
  virtual bool get_use_kl_info()
  {
	  return kl_info_vec.size() > 0;
  }
  virtual int label(){ return label_list[key_index_list[key_pos-1]]; }
  virtual vector<float>& kl_info(){ return  kl_info_vec[key_index_list[key_pos - 1]]; }
  virtual void set_shuffle(bool val){ is_shuffle = val; }
  virtual void set_extra_data_type(DataParameter::DataType extra_data_type){ extra_data_type_ = extra_data_type; }
  virtual DataParameter::DataType get_extra_data_type(){ return extra_data_type_; }
protected:
	vector<int> key_index_list;
	vector<string> key_list;
	vector<int> label_list;
	vector<vector<float>> kl_info_vec;
	size_t key_pos = 0;
	void get_key_from_db();

	bool valid_;
	bool is_shuffle;

	DataParameter::DataType extra_data_type_ = DataParameter_DataType_NULL_DATA;
  DISABLE_COPY_AND_ASSIGN(Cursor);
};

class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_AND_ASSIGN(Transaction);
};

class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;
  virtual Transaction* NewTransaction() = 0;
  DISABLE_COPY_AND_ASSIGN(DB);
};

DB* GetDB(DataParameter::DB backend);
DB* GetDB(const string& backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_HPP
