#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/binary_bounding_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class BinaryBoundingLossTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
	 BinaryBoundingLossTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 3, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(5, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
	 virtual ~BinaryBoundingLossTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BinaryBoundingLossTest, TestDtypesAndDevices);

TYPED_TEST(BinaryBoundingLossTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.mutable_binary_bounding_param()->set_alpha(0.2);
  layer_param.mutable_binary_bounding_param()->set_beta(0.4);
  layer_param.mutable_binary_bounding_param()->set_threshold(4);
  layer_param.mutable_binary_bounding_param()->set_ratio(0.9);
  layer_param.mutable_cluster_centroid_param()->mutable_centroid_filler()->set_type("gaussian");

  BinaryBoundingLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  CHECK_EQ(1,0) << "loss " << this->blob_top_vec_[0]->cpu_data()[0] << "," << layer.blobs()[0]->cpu_data()[0] << "," << layer.blobs()[0]->cpu_data()[1] << "," << layer.blobs()[0]->cpu_data()[2];
}
 
}  // namespace caffe
