#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/cluster_centroid_dist_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_softmax_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename TypeParam>
class CentroidDistTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
	 CentroidDistTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
		 Dtype bottom_[] = {1,2,1,0,0,2};
		 memcpy(blob_bottom_->mutable_cpu_data(), bottom_, blob_bottom_->count()*sizeof(Dtype));
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
	 virtual ~CentroidDistTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CentroidDistTest, TestDtypesAndDevices);

TYPED_TEST(CentroidDistTest, TestForward) { 
  typedef typename TypeParam::Dtype Dtype;
  Dtype bottom_[] = { 1, 2, 1, 0, 0, 2 };
  memcpy(blob_bottom_->mutable_cpu_data(), bottom_, blob_bottom_->count()*sizeof(Dtype));
  LayerParameter layer_param;
  layer_param.mutable_cluster_centroid_param()->set_num_cluster(2);
  layer_param.mutable_cluster_centroid_param()->set_dim(3);

  ClusterCentroidDistLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype centroid[] = { 0, 0, 1, 1, 0, 2 };
  memcpy(layer.blobs()[0]->mutable_cpu_data(), centroid, layer.blobs()[0]->count()*sizeof(Dtype));
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  for (int i = 0; i < layer.blobs()[0]->count(); ++i)
	  EXPECT_EQ(layer.blobs()[0]->cpu_data()[i], centroid[i]);

  for (int i = 0; i < layer.blobs()[1]->count();++i)
	  EXPECT_EQ(layer.blobs()[1]->cpu_data()[i], 1);

  Dtype result[] = { 2.5, 2.5, 0.5, 0.5 };
  for (int i = 0; i < this->blob_top_->count(); ++i) 
	  EXPECT_EQ(this->blob_top_->cpu_data()[i], result[i]) <<
	  "not equal: " << this->blob_top_->cpu_data()[i] << " vs " << result[i];
}

TYPED_TEST(CentroidDistTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_cluster_centroid_param()->set_num_cluster(2);
  layer_param.mutable_cluster_centroid_param()->set_dim(3);
  SoftmaxLayer<Dtype> layer(layer_param);
  //Dtype centroid[] = { 0, 0, 1, 1, 0, 2 };
	//memcpy(layer.blobs()[0]->mutable_cpu_data(), centroid, layer.blobs()[0]->count()*sizeof(Dtype));

  layer_param.mutable_cluster_centroid_param()->mutable_centroid_filler()->set_type("gaussian");
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
