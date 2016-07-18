#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"
#include "caffe/util/db_base64.hpp"
#include <string>
#include <ctime>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case DataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
  case DataParameter_DB_BASE64:
	  return new Base64();
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(const string& backend) {
#ifdef USE_LEVELDB
  if (backend == "leveldb") {
    return new LevelDB();
  }
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  if (backend == "lmdb") {
    return new LMDB();
  }
#endif  // USE_LMDB
  if (backend == "base64")
  {
	  return new Base64();
  }
  LOG(FATAL) << "Unknown database backend";
  return NULL;
}

void Cursor::set_key_by_file(string key_file_name)
{
	LOG(INFO) << "set key by key_file: " << key_file_name;
	key_list.clear();
	label_list.clear();
	kl_info_vec.clear();
	key_index_list.clear();

	std::ifstream key_file(key_file_name);
	string key;
	int label;
	//while (key_file >> key >>label)
	string line;
	while (std::getline(key_file, line))
	{
		vector<string> str_vec = string_split(line, ' ');
		key = str_vec[0];
		label = atoi(str_vec[1].c_str());

		key_list.push_back(key);
		label_list.push_back(label);

		if (str_vec.size() >= 3)
		{
			vector<float> tmp_info;
			for (int info_idx = 2; info_idx < str_vec.size(); ++info_idx)
			{
				tmp_info.push_back(std::stof(str_vec[info_idx]));
			}
			kl_info_vec.push_back(tmp_info);
		}
	}

	for (int i = 0; i < key_list.size(); ++i)
	{
		key_index_list.push_back(i);
	}
	LOG(INFO) << "data size: " << key_index_list.size();
}

void Cursor::get_key_from_db()
{
	LOG(INFO) << "set key from db ";
	key_list.clear();
	label_list.clear();
	kl_info_vec.clear();
	this->SeekToFirst();
	while (this->valid())
	{
		string key = this->key();
		key_list.push_back(key);
		this->Next();
	}

	this->SeekToFirst();
	for (int i = 0; i < key_list.size(); ++i)
	{
		key_index_list.push_back(i);
	}
	LOG(INFO) << "data size: " << key_index_list.size();
}

void Cursor::shuffle()
{
	if (key_index_list.size() == 0)
		get_key_from_db();
	std::srand(std::time(0));
	LOG(INFO) << "shuffle data base";
	std::random_shuffle(key_index_list.begin(), key_index_list.end());
}

}  // namespace db
}  // namespace caffe
