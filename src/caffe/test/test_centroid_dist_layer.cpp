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
  Blob<Dtype>*  blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CentroidDistTest, TestDtypesAndDevices);

TYPED_TEST(CentroidDistTest, TestForward) { 
  typedef typename TypeParam::Dtype Dtype;
  Dtype bottom_[] = { 1, 2, 1, 0, 0, 2 };
  memcpy(this->blob_bottom_->mutable_cpu_data(), bottom_, this->blob_bottom_->count()*sizeof(Dtype));
  LayerParameter layer_param;
  layer_param.mutable_cluster_centroid_param()->set_num_cluster(2);
  layer_param.mutable_cluster_centroid_param()->set_dim(3);
  layer_param.mutable_binary_bounding_param()->set_not_initialed(false);
  layer_param.set_phase(TEST);

  ClusterCentroidDistLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  Dtype centroid[] = { 0, 0, 1, 1, 0, 2 };
  memcpy(layer.blobs()[0]->mutable_cpu_data(), centroid, layer.blobs()[0]->count()*sizeof(Dtype));
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  for (int i = 0; i < layer.blobs()[0]->count(); ++i)
	  EXPECT_EQ(layer.blobs()[0]->cpu_data()[i], centroid[i]);

  //for (int i = 0; i < layer.blobs()[1]->count();++i)
	 // EXPECT_EQ(layer.blobs()[1]->cpu_data()[i], 1);

  Dtype result[] = { sqrt(5.0 / 2.0), sqrt(5.0 / 2.0), sqrt(1.0 / 2.0), sqrt(1.0 / 2.0) };
  for (int i = 0; i < this->blob_top_->count(); ++i) 
	  EXPECT_NEAR(this->blob_top_->cpu_data()[i], result[i],1e-4) <<
	  "not equal: " << this->blob_top_->cpu_data()[i] << " vs " << result[i];
}

TYPED_TEST(CentroidDistTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_ = new Blob<Dtype>(10, 3, 1, 1);
  FillerParameter filler_param;
  filler_param.set_std(1);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  this->blob_bottom_vec_[0] = this->blob_bottom_;
  LayerParameter layer_param;
  layer_param.mutable_cluster_centroid_param()->set_num_cluster(3);
  layer_param.mutable_cluster_centroid_param()->set_dim(3);
  layer_param.mutable_cluster_centroid_param()->set_scale(-10);
  layer_param.mutable_softmax_param()->set_temperature(100);
  layer_param.mutable_binary_bounding_param()->set_not_initialed(false);
  layer_param.set_phase(TEST); 
  layer_param.mutable_cluster_centroid_param()->mutable_centroid_filler()->set_type("gaussian");
  ClusterCentroidDistLayer<Dtype> layer(layer_param);
  //Dtype centroid[] = { 0, 0, 1, 1, 0, 2 };
	//memcpy(layer.blobs()[0]->mutable_cpu_data(), centroid, layer.blobs()[0]->count()*sizeof(Dtype));
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
