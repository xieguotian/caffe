"../../../3rdparty/bin/protoc" caffe.proto --cpp_out="./"
"../../../3rdparty/bin/protoc" caffe.proto --python_out="./"
copy caffe_pb2.py "../../../python/caffe/proto/"