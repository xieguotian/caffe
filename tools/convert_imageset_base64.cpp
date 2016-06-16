// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#define snprintf sprintf_s

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
	"When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
	"Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
	"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
	"When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
	"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
	"Optional: What type should we encode the image as ('png','jpg',...).");


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


int main(int argc, char** argv) {
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
		"The ImageNet dataset for the training demo is at\n"
		"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
		return 1;
	}

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;

	std::ifstream infile(argv[2]);
	std::vector<std::pair<std::string, int> > lines;
	std::string line;
	size_t pos;
	int label;
	/*
	while (std::getline(infile, line)) {
	pos = line.find_last_of(' ');
	label = atoi(line.substr(pos + 1).c_str());
	lines.push_back(std::make_pair(line.substr(0, pos), label));
	}*/

	string name = argv[2];
	pos = name.find_last_of('.');
	string re_name = name.substr(0, pos) + ".label";
	std::ifstream label_file(re_name);
	while (std::getline(label_file, line))
	{
		pos = line.find_last_of(' ');
		label = atoi(line.substr(pos + 1).c_str());
		lines.push_back(std::make_pair(line.substr(0, pos), label));
	}

	LOG(INFO) << "A total of " << lines.size() << " images.";

	if (encode_type.size() && !encoded)
		LOG(INFO) << "encode_type specified, assuming encoded=true.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[3], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	std::string root_folder(argv[1]);
	Datum datum;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;

	for (int line_id = 0; line_id < lines.size(); ++line_id) {

		string tmp_line;
		std::getline(infile, tmp_line);
		pos = tmp_line.find_last_of('\t');
		string key = tmp_line.substr(0, pos);
		string img_base64 = tmp_line.substr(pos + 1);
		CHECK(key == lines[line_id].first);

		bool status;
		// Guess the encoding type from the file name
		string fn = lines[line_id].first;

		string buffer = base64_decode(img_base64);
		status = SetBufferToDatum(buffer, label, &datum);
		if (status == false) continue;

		// sequential
		string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
