#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include <cuda_fp16.h>

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0),can_be_save_as_bin_(false),is_binarized_(false) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0), can_be_save_as_bin_(false),is_binarized_(false) {
  Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareData_LE(const Blob& other) {
	CHECK_LE(count_, other.count());
	data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff_LE(const Blob& other) {
	CHECK_LE(count_, other.count());
	diff_ = other.diff();
}

// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<char>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<unsigned char>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> char Blob<char>::asum_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <> unsigned char Blob<unsigned char>::asum_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> char Blob<char>::asum_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <> unsigned char Blob<unsigned char>::asum_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> char Blob<char>::sumsq_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <> unsigned char Blob<unsigned char>::sumsq_data() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> char Blob<char>::sumsq_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <> unsigned char Blob<unsigned char>::sumsq_diff() const {
	NOT_IMPLEMENTED;
	return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<char>::scale_data(char scale_factor) {
	NOT_IMPLEMENTED;
}

template <> void Blob<unsigned char>::scale_data(unsigned char scale_factor) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<char>::scale_diff(char scale_factor) {
	NOT_IMPLEMENTED;
}

template <> void Blob<unsigned char>::scale_diff(unsigned char scale_factor) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape, bool force_copy) {
	if ((source.count() != count_ || source.shape() != shape_) && !force_copy) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  int copy_count = count_;
  if (force_copy)
  {
	  copy_count = std::min(count_, source.count());
	  LOG(INFO) << "force copy from source to target (" 
		  << source.count() << " vs " << count_ << ")";
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(copy_count, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(copy_count, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(copy_count, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(copy_count, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
  if (reshape) {
    vector<int> shape;
    if (proto.has_num() || proto.has_channels() ||
        proto.has_height() || proto.has_width()) {
      // Using deprecated 4D Blob dimensions --
      // shape is (num, channels, height, width).
      shape.resize(4);
      shape[0] = proto.num();
      shape[1] = proto.channels();
      shape[2] = proto.height();
      shape[3] = proto.width();
    } else {
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape[i] = proto.shape().dim(i);
      }
    }
    Reshape(shape);
  } else {
    CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
  }
  is_binarized_ = proto.is_binarized();
  if (is_binarized_)
  {
	  can_be_save_as_bin_ = is_binarized_;
	  Dtype* data_vec = mutable_cpu_data();
	  bool is_double = proto.double_data_size() > 0;
	  if (is_double)
		  CHECK_EQ(num(), proto.double_data_size());
	  else
		  CHECK_EQ(num(), proto.data_size());

	  string bin_data = proto.bin_data();

	  if (bin_data.size() > 0)
	  {
		  int spat_dim = channels()*height()*width();
		  int num_bytes = std::ceil(count_ / 8.0);
		  CHECK_EQ(num_bytes, bin_data.size());
		  std::vector<char> bin_vec_data(bin_data.c_str(), bin_data.c_str() + bin_data.size());

		  int idx_bit = 0;
		  for (int i = 0; i < num_bytes; ++i){
			  char val = bin_vec_data[i];
			  for (int j = 0; j < 8; j++)
			  {
				  if (idx_bit < count_)
				  {
					  int num_idx = idx_bit / spat_dim;
					  Dtype ampl_val;
					  Dtype bin_val;
					  if (is_double)
					  {
						  ampl_val = proto.double_data(num_idx);
					  }
					  else{
						  ampl_val = proto.data(num_idx);
					  }
					  bin_val = (val & 0x80) > 0 ? 1 : -1;
					  val = val << 1;
					  data_vec[idx_bit] = bin_val*ampl_val;
					  idx_bit++;
				  }
			  }
		  }
	  }
  }
  else
  {
	  // copy data
	  Dtype* data_vec = mutable_cpu_data();
	  if (proto.double_data_size() > 0) {
		  CHECK_EQ(count_, proto.double_data_size());
		  for (int i = 0; i < count_; ++i) {
			  data_vec[i] = proto.double_data(i);
		  }
	  }
	  else {
		  CHECK_EQ(count_, proto.data_size());
		  for (int i = 0; i < count_; ++i) {
			  data_vec[i] = proto.data(i);
		  }
	  }
	  if (proto.double_diff_size() > 0) {
		  CHECK_EQ(count_, proto.double_diff_size());
		  Dtype* diff_vec = mutable_cpu_diff();
		  for (int i = 0; i < count_; ++i) {
			  diff_vec[i] = proto.double_diff(i);
		  }
	  }
	  else if (proto.diff_size() > 0) {
		  CHECK_EQ(count_, proto.diff_size());
		  Dtype* diff_vec = mutable_cpu_diff();
		  for (int i = 0; i < count_; ++i) {
			  diff_vec[i] = proto.diff(i);
		  }
	  }
  }
}


template <>
void Blob<float>::to_bin(Blob<float> &bin, Blob<float> &ampl) const
{
	bin.ReshapeLike(*this);
	vector<int> shape_tmp;
	shape_tmp.push_back(bin.num());
	ampl.Reshape(shape_tmp);
	Blob<float> ones;

	shape_tmp[0] = channels()*height()*width();
	ones.Reshape(shape_tmp);
	caffe_gpu_set(ones.count(), (float)1.0, ones.mutable_gpu_data());

	// binarize weight
	caffe_gpu_sign(count_, gpu_data(), bin.mutable_gpu_data());
	// calculate abs(weight)
	caffe_gpu_abs(count_, gpu_data(), bin.mutable_gpu_diff());

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
		num(), 1,
		channels()*height()*width(),
		(float)(1.0 / num()),
		//this->blobs_[0]->gpu_data(),
		bin.gpu_diff(),
		ones.gpu_data(),
		(float)0.0,
		ampl.mutable_gpu_data()
		);
}
template <>
void Blob<double>::to_bin(Blob<double> &bin, Blob<double> &ampl) const
{
	bin.ReshapeLike(*this);
	vector<int> shape_tmp;
	shape_tmp.push_back(bin.num());
	ampl.Reshape(shape_tmp);
	Blob<double> ones;

	shape_tmp[0] = channels()*height()*width();
	ones.Reshape(shape_tmp);
	caffe_gpu_set(ones.count(), (double)1.0, ones.mutable_gpu_data());

	// binarize weight
	caffe_gpu_sign(count_, gpu_data(), bin.mutable_gpu_data());
	// calculate abs(weight)
	caffe_gpu_abs(count_, gpu_data(), bin.mutable_gpu_diff());

	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
		num(), 1,
		channels()*height()*width(),
		(double)(1.0 / num()),
		//this->blobs_[0]->gpu_data(),
		bin.gpu_diff(),
		ones.gpu_data(),
		(double)0.0,
		ampl.mutable_gpu_data()
		);
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff, bool save_as_bin) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  if (save_as_bin && can_be_save_as_bin_)
  {
#ifndef CPU_ONLY
	  proto->clear_bin_data();
	  Blob<double> bin_data;
	  Blob<double> ampl_data;
	  to_bin(bin_data, ampl_data);
	  const double* bin_data_vec = bin_data.cpu_data();
	  const double* ampl_data_vec = ampl_data.cpu_data();
	  int num_bytes = std::ceil(count_ / 8.0);
	  int idx_bit = 0;
	  char* all_bin_data = new char[num_bytes];
	  for (int i = 0; i < num_bytes; i++)
	  {

		  char val = 0;
		  for (int j = 0; j < 8; j++)
		  {
			  if (idx_bit < count_)
			  {
				  char bit_val = bin_data_vec[idx_bit]>0 ? 1 : 0;
				  val = val << 1 | bit_val;
				  idx_bit++;
			  }
			  else
			  {
				  val = val << 1;
			  }
		  }
		  all_bin_data[i] = val;
	  }
	  proto->set_bin_data(all_bin_data, num_bytes);
	  for (int i = 0; i < ampl_data.count(); i++)
	  {
		  proto->add_double_data(ampl_data_vec[i]);
	  }
	  proto->set_is_binarized(true);
#elif
	  NOT_IMPLEMENTED
#endif
  }
  else{
	  for (int i = 0; i < count_; ++i) {
		  proto->add_double_data(data_vec[i]);
	  }
	  if (write_diff) {
		  const double* diff_vec = cpu_diff();
		  for (int i = 0; i < count_; ++i) {
			  proto->add_double_diff(diff_vec[i]);
		  }
	  }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff, bool save_as_bin) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
  const float* data_vec = cpu_data();
  if (save_as_bin && can_be_save_as_bin_)
  {
#ifndef CPU_ONLY
	  proto->clear_bin_data();
	  Blob<float> bin_data;
	  Blob<float> ampl_data;
	  to_bin(bin_data, ampl_data);
	  const float* bin_data_vec = bin_data.cpu_data();
	  const float* ampl_data_vec = ampl_data.cpu_data();

	  int num_bytes = std::ceil(count_ / 8.0);
	  int idx_bit = 0;
	  char* all_bin_data = new char[num_bytes];

	  for (int i = 0; i < num_bytes; i++)
	  {
		  char val = 0;
		  for (int j = 0; j < 8; j++)
		  {
			  if (idx_bit < count_)
			  {
				  char bit_val = bin_data_vec[idx_bit]>0 ? 1 : 0;
				  val = val << 1 | bit_val;
				  idx_bit++;
			  }
			  else
			  {
				  val = val << 1;
			  }
		  }
		  all_bin_data[i] = val;
	  }
	  proto->set_bin_data(all_bin_data, num_bytes);

	  for (int i = 0; i < ampl_data.count(); i++)
	  {
		  proto->add_data(ampl_data_vec[i]);
	  }
	  proto->set_is_binarized(true);
	  //LOG(INFO)<<"to binary: " << count_;
#elif
	  NOT_IMPLEMENTED
#endif
  }
  else{
	  for (int i = 0; i < count_; ++i) {
		  proto->add_data(data_vec[i]);
	  }
	  if (write_diff) {
		  const float* diff_vec = cpu_diff();
		  for (int i = 0; i < count_; ++i) {
			  proto->add_diff(diff_vec[i]);
		  }
	  }
	  //LOG(INFO) << "to float: "<<count_;
  }
}



template <>
void Blob<char>::to_bin(Blob<char> &bin, Blob<char> &ampl)  const
{
	NOT_IMPLEMENTED;
}
template <>
void Blob<unsigned char>::to_bin(Blob<unsigned char> &bin, Blob<unsigned char> &ampl)  const
{
	NOT_IMPLEMENTED;
}
template <>
void Blob<int>::to_bin(Blob<int> &bin, Blob<int> &ampl) const
{
	NOT_IMPLEMENTED;
}
template <>
void Blob<unsigned int>::to_bin(Blob<unsigned int> &bin, Blob<unsigned int> &ampl) const
{
	NOT_IMPLEMENTED;
}
//template <>
//void Blob<double>::to_bin(Blob<double> &bin, Blob<double> &ampl)
//{
//	bin->ReshapeLike(*this);
//	vector<int> shape_tmp;
//	shape_tmp.push_back(bin->num());
//	ampl->Reshape(shape_tmp);
//	Blob<double> ones;
//
//	shape_tmp[0] = channels()*height()*width();
//	ones.Reshape(shape_tmp);
//	caffe_gpu_set<double>(ones.count(), (double)1.0, ones.mutable_gpu_data());
//
//	// binarize weight
//	caffe_gpu_sign<double>(count_, gpu_data(), bin->mutable_gpu_data());
//	// calculate abs(weight)
//	caffe_gpu_abs<double>(count_, gpu_data(), bin->mutable_gpu_diff());
//
//	caffe_gpu_gemm<double>(CblasNoTrans, CblasNoTrans,
//		num(), 1,
//		channels()*height()*width(),
//		(double)1.0 / num(),
//		//this->blobs_[0]->gpu_data(),
//		bin->gpu_diff(),
//		ones.gpu_data(),
//		(double)0.0,
//		ampl->mutable_gpu_data()
//		);
//}
INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;
template class Blob<char>;
template class Blob<unsigned char>;
//template class Blob<half>;

}  // namespace caffe

