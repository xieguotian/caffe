#ifndef CAFFE_UTIL_DB_BASE64_HPP
#define CAFFE_UTIL_DB_BASE64_HPP

#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe {
	namespace db {
		static const std::string base64_chars =
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"abcdefghijklmnopqrstuvwxyz"
			"0123456789+/";

		static inline bool is_base64(unsigned char c) {
			return (isalnum(c) || (c == '+') || (c == '/'));
		}

		string base64_decode(string const& encoded_string) {
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

		string base64ToDatumString(string base64_str)
		{
			string img_label = base64_str;
			int pos = img_label.find_last_of(' '); 
			if (pos == string::npos)
				pos = img_label.find_last_of('\t');
			int label;
			string img;
			if (pos == string::npos)
			{
				label = 0;
				img = base64_decode(img_label);
			}
			else
			{
				label = atoi(img_label.substr(pos + 1).c_str());
				img = base64_decode(img_label.substr(0, pos));
			}
			Datum datum;
			datum.set_data(img);
			datum.set_label(label);
			datum.set_encoded(true);

			string result;
			datum.SerializeToString(&result);
			return result;
		}

		class Base64Cursor : public Cursor {
		public:
			explicit Base64Cursor( string base64_cursor)
				: valid_(false), is_key_pos_map_ready_(false){
				base64_cursor_.open(base64_cursor);
				SeekToFirst();
			}
			virtual ~Base64Cursor() {
				close();
			}
			void close(){
				base64_cursor_.close();
			}
			virtual void SeekToFirst() {
				base64_cursor_.clear();
				base64_cursor_.seekg(0);
				string line;
				if (std::getline(base64_cursor_, line))
				{
					int pos = line.find_first_of('\t');
					if (pos == string::npos)
					{
						valid_ = false;
						LOG(INFO) << "first split error";
						return;
					}
					base64_key_ = line.substr(0, pos);
					base64_value_ = base64ToDatumString(line.substr(pos + 1));
					valid_ = true;
				}
				else{
					valid_ = false;
					LOG(INFO) << "first no line";
				}
			}
			void init_key_pos_map()
			{
				LOG(INFO) << "begin initial base64 key-pos map.";
				string line;
				size_t line_pos = base64_cursor_.beg;
				while (std::getline(base64_cursor_, line))
				{
					int pos = line.find_first_of('\t');
					string key = line.substr(0, pos);
					key_pos_map_[key] = line_pos;
					line_pos = base64_cursor_.tellg();
				}
				base64_cursor_.clear();
				base64_cursor_.seekg(0);
				LOG(INFO) << "end initial base64 key-pos map.";
			}
			virtual void Next() {
				string line;
				if (std::getline(base64_cursor_, line))
				{
					int pos = line.find_first_of('\t');
					if (pos == string::npos)
					{
						valid_ = false;
						LOG(INFO) << "next split error";
						return;
					}
					base64_key_ = line.substr(0, pos);
					base64_value_ = base64ToDatumString(line.substr(pos + 1));
					valid_ = true;
				}
				else
				{ 
					//SeekToFirst();
					valid_ = false;
					LOG(INFO) << "next no line";
				}
			}

			virtual string key() {
				return base64_key_;
			}
			virtual string value() {
				return base64_value_;
			}

			virtual bool valid() { return valid_; } 

			virtual void SeekByKey(string key)
			{
				if (!is_key_pos_map_ready_)
				{
					init_key_pos_map();
					is_key_pos_map_ready_ = true;
				}
				if (base64_cursor_.eof())
				{
					base64_cursor_.clear();
					base64_cursor_.seekg(0);
				}
				if (key_pos_map_.find(key) == key_pos_map_.end())
				{
					valid_ = false;
					LOG(INFO) << "key not find: " << key_pos_map_.size();
					return;
				}
				size_t line_pos = key_pos_map_[key];
				base64_cursor_.seekg(line_pos);
				string line;
				if (std::getline(base64_cursor_, line))
				{
					int pos = line.find_first_of('\t');
					if (pos == string::npos)
					{
						LOG(INFO) << "read key fail: "  << line << "\t" << key;
						valid_ = false;
						return;
					}
					base64_key_ = line.substr(0, pos);
					base64_value_ = base64ToDatumString(line.substr(pos + 1));
					valid_ = base64_key_ == key;
					if (!valid_)
						LOG(INFO) << "key1:" << base64_key_ << "," << key;
				}
				else
				{
					valid_ = false;
					LOG(INFO) << "read key fail: " << key;
				}
			}
		private:
			string base64_key_, base64_value_;
			map<string, size_t> key_pos_map_;
			bool valid_;
			std::ifstream base64_cursor_;
			bool is_key_pos_map_ready_;
		};

		class Baset64Transaction : public Transaction {
		public:
			explicit Baset64Transaction() {}
			virtual void Put(const string& key, const string& value){}
			virtual void Commit(){}
		};

		class Base64 : public DB {
		public:
			Base64(){}
			virtual ~Base64() { Close(); }
			virtual void Open(const string& source, Mode mode)
			{
				base64_stream_ = source;
				LOG(INFO) << "begin opening bas64 " << source;
				base64_cursor = new Base64Cursor(base64_stream_);
				LOG(INFO) << "Opened base64 " << source;
			}
			virtual void Close() {
				base64_cursor->close();
			}

			virtual Base64Cursor* NewCursor()
			{
				return base64_cursor;
			}

			virtual Baset64Transaction* NewTransaction(){ return new Baset64Transaction(); }

		private:
			string base64_stream_;
			Base64Cursor* base64_cursor;
		};

	}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
