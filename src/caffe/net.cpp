#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const Net* root_net)
    : root_net_(root_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  param.mutable_state()->set_phase(phase);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  CHECK(Caffe::root_solver() || root_net_)
      << "root_net_ needs to be set for all non-root solvers";
  // Set phase from the state.
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  FilterNet(in_param, &filtered_param);
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  memory_used_ = 0;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // For non-root solvers, whether this layer is shared from root_net_.
    bool share_from_root = !Caffe::root_solver()
        && root_net_->layers_[layer_id]->ShareInParallel();
    // Inherit phase from net if unset.
    if (!param.layer(layer_id).has_phase()) {
      param.mutable_layer(layer_id)->set_phase(phase_);
    }
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    if (layer_param.propagate_down_size() > 0) {
      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }
    if (share_from_root) {
      LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
      layers_.push_back(root_net_->layers_[layer_id]);
      layers_[layer_id]->SetShared(true);
    } else {
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    }
    layer_names_.push_back(layer_param.name());
	layer_types_.push_back(layer_param.type());
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    Layer<Dtype>* layer = layers_[layer_id].get();
    if (layer->AutoTopBlobs()) {
      const int needed_num_top =
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    if (share_from_root) {
      // Set up size of top blobs using root_net_
      const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
      const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
      for (int top_id = 0; top_id < base_top.size(); ++top_id) {
        this_top[top_id]->ReshapeLike(*base_top[top_id]);
        LOG(INFO) << "Created top blob " << top_id << " (shape: "
            << this_top[top_id]->shape_string() <<  ") for shared layer "
            << layer_param.name();
      }
    } else {
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    }
	
	//if this is not a loss layer then a blob is deletable
	bool is_deletable = layers_[layer_id]->layer_param().loss_weight_size() == 0;
	for (int ti = 0; ti < top_vecs_[layer_id].size(); ++ti)
	{
		const int blob_id = top_id_vecs_[layer_id][ti];
		blobs_deletable_[blob_id] = is_deletable;
		if (!is_deletable)
			LOG(INFO) << "layer " << layer_names_[layer_id] << " top_id " << ti << " :" << "undeletabble";
	}

    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);
    const int param_size = layer_param.param_size();
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    ParamSpec default_param_spec;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      const ParamSpec* param_spec = (param_id < param_size) ?
          &layer_param.param(param_id) : &default_param_spec;
      const bool param_need_backward = param_spec->lr_mult() != 0;
      need_backward |= param_need_backward;
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  set<string> blobs_under_loss;
  set<string> blobs_skip_backp;
  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed.
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();

  opt_memory_ = param.opt_memory();
  opt_test_shared_memory_ = param.opt_test_shared_memory();
  shared_blobs_.clear();
  shared_record_.clear();
  if (phase_ == Phase::TEST && opt_test_shared_memory_ || phase_ == Phase::TRAIN && opt_memory_)
  {
	  shared_blobs_index_.resize(blobs_.size());
	  for (int i = 0; i < layers_.size(); ++i) 
	  {
		  if ((phase_ == Phase::TEST && opt_test_shared_memory_) || (phase_ == Phase::TRAIN && opt_memory_&& !layer_need_backward_[i]))
		  {
			  if (layer_types_[i] == "Split")
			  {
				  //don't append shared memory for top, but increase the shared_record_
				  //release shared memory for bottom.
				  const int blob_id = bottom_id_vecs_[i][0];
				  const int shared_blob_id = shared_blobs_index_[blob_id];

				  shared_record_[shared_blob_id] -= 1;


				  //append shared memory for top.
				  for (int ti = 0; ti < top_vecs_[i].size(); ++ti)
				  {
					  const int blob_id_top = top_id_vecs_[i][ti];
					  shared_record_[shared_blob_id] += 1;
					  shared_blobs_index_[blob_id_top] = shared_blobs_index_[blob_id];
				  }
				  continue;
			  }


			  //append shared memory for top.
			  for (int ti = 0; ti < top_vecs_[i].size(); ++ti)
			  {
				  // if in-place, don't borrow the shared memory again.
				  if (bottom_vecs_[i].size()>ti && bottom_vecs_[i][ti] == top_vecs_[i][ti])
					  continue;

				  bool is_find_cache = false;
				  int cache_idx = -1;
				  const int blob_id = top_id_vecs_[i][ti];

				  if (blob_used_counter_[blob_id] == 1)
				  {
					  shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
					  shared_blobs_.push_back(blob_pointer);
					  cache_idx = shared_blobs_.size() - 1;

					  shared_record_.push_back(blob_used_counter_[blob_id]);
				  }
				  else
				  {
					  //search for cache memory.
					  for (int si = 0; si < shared_blobs_.size(); si++)
					  {
						  if (shared_record_[si] == 0)
						  {
							  //found cache
							  is_find_cache = true;
							  cache_idx = si;
							  shared_record_[si] = blob_used_counter_[blob_id];
							  break;
						  }
					  }
					  //if not found cache, create new cache.
					  if (!is_find_cache)
					  {
						  shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
						  shared_blobs_.push_back(blob_pointer);
						  cache_idx = shared_blobs_.size() - 1;

						  shared_record_.push_back(blob_used_counter_[blob_id]);
					  }

				  }

				  //share cache meory.
				  if (shared_blobs_[cache_idx]->count() < blobs_[blob_id]->count())
				  {
					  shared_blobs_[cache_idx]->ReshapeLike(*blobs_[blob_id]);
					  LOG_IF(INFO, Caffe::root_solver()) << "reshape " << layer_types_[i] << " " << shared_blobs_[cache_idx]->count();
				  }
				  blobs_[blob_id]->ShareData_LE(*shared_blobs_[cache_idx]);
				  shared_blobs_index_[blob_id] = cache_idx;
			  }

			  //release shared memory for bottom.
			  for (int bi = 0; bi < bottom_vecs_[i].size(); ++bi)
			  {
				  const int blob_id = bottom_id_vecs_[i][bi];
				  const int shared_blob_id = shared_blobs_index_[blob_id];

				  shared_record_[shared_blob_id] -= 1;
				  //release memory
				  if (shared_record_[shared_blob_id] == 1)
					  shared_record_[shared_blob_id] = 0;
			  }
		  }
	  }
	  LOG_IF(INFO, Caffe::root_solver()) << "cache size for test net is : " << shared_blobs_.size();
	  for (int i = 0; i < shared_blobs_.size(); i++)
		  LOG_IF(INFO, Caffe::root_solver()) << "cache id " << i << " size:" << shared_blobs_[i]->count();
  }
  if (phase_ == Phase::TRAIN && opt_memory_)
  {
	  blob_diff_used_counter_.resize(blobs_.size(),0);

	  for (int layer_id = layers_.size() - 1; layer_id >= 0; layer_id--)
	  {
		  for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size(); ++bottom_id)
		  {
			  if (bottom_id < top_vecs_[layer_id].size() && bottom_vecs_[layer_id][bottom_id] == top_vecs_[layer_id][bottom_id])
				  continue;
			  int blob_id = bottom_id_vecs_[layer_id][bottom_id];
			  blob_diff_used_counter_[blob_id] = 1;
		  }

		  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id)
		  {
			  const int blob_id = top_id_vecs_[layer_id][top_id];

			  if (blob_used_counter_[blob_id] == 1)
				  continue;

			  blob_diff_used_counter_[blob_id] += 1;
		  }
	  }
	  //shared memory
	  shared_blobs_diff_index_.resize(blobs_.size());
	  for (int layer_id = layers_.size() - 1; layer_id >= 0; layer_id--)
	  {
		  if (phase_ == Phase::TRAIN && opt_memory_)//&& layer_need_backward_[i])
		  {

			  //append shared memory for top.
			  for (int bi = 0; bi < bottom_vecs_[layer_id].size(); ++bi)
			  {
				  // if in-place, don't borrow the shared memory again.
				  if (top_vecs_[layer_id].size()>bi && top_vecs_[layer_id][bi] == bottom_vecs_[layer_id][bi])
					  continue;

				  bool is_find_cache = false;
				  int cache_idx = -1;
				  const int blob_id = bottom_id_vecs_[layer_id][bi];

				  if (blob_diff_used_counter_[blob_id] == 1)
				  {
					  shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
					  shared_blobs_diff_.push_back(blob_pointer);
					  cache_idx = shared_blobs_diff_.size() - 1;

					  shared_record_diff_.push_back(blob_diff_used_counter_[blob_id]);
				  }
				  else
				  {
					  //search for cache memory.
					  for (int si = 0; si < shared_record_diff_.size(); si++)
					  {
						  if (shared_record_diff_[si] == 0)
						  {
							  //found cache
							  is_find_cache = true;
							  cache_idx = si;
							  shared_record_diff_[si] = blob_diff_used_counter_[blob_id];
							  break;
						  }
					  }
					  //if not found cache, create new cache.
					  if (!is_find_cache)
					  {
						  shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
						  shared_blobs_diff_.push_back(blob_pointer);
						  cache_idx = shared_blobs_diff_.size() - 1;

						  shared_record_diff_.push_back(blob_diff_used_counter_[blob_id]);
					  }

				  }

				  //share cache meory.
				  if (shared_blobs_diff_[cache_idx]->count() < blobs_[blob_id]->count())
				  {
					  shared_blobs_diff_[cache_idx]->ReshapeLike(*blobs_[blob_id]);
					  LOG_IF(INFO, Caffe::root_solver()) << "reshape diff of " << layer_types_[layer_id] << " " << shared_blobs_diff_[cache_idx]->count();
				  }
				  blobs_[blob_id]->ShareDiff_LE(*shared_blobs_diff_[cache_idx]);
				  shared_blobs_diff_index_[blob_id] = cache_idx;
			  }

			  //release shared memory for bottom.
			  for (int ti = 0; ti < top_vecs_[layer_id].size(); ++ti)
			  {
				  const int blob_id = top_id_vecs_[layer_id][ti];
				  const int shared_blob_id = shared_blobs_diff_index_[blob_id];

				  shared_record_diff_[shared_blob_id] -= 1;
				  //release memory
				  if (shared_record_diff_[shared_blob_id] == 1)
					  shared_record_diff_[shared_blob_id] = 0;
			  }
		  }
	  }
	  LOG_IF(INFO, Caffe::root_solver()) << "diff cache size for train net is : " << shared_blobs_diff_.size();
	  for (int i = 0; i < shared_blobs_diff_.size(); i++)
		  LOG_IF(INFO, Caffe::root_solver()) << "cache id " << i << " size:" << shared_blobs_diff_[i]->count();
	  /*
	  // init mem optimization
	  for (int type_id = 0; type_id < layer_types_.size(); ++type_id)
	  {
		  if (layer_types_[type_id] == "Convolution" || layer_types_[type_id] == "BatchNorm"
			  || layer_types_[type_id] == "BatchNormOpt"
			  || layer_types_[type_id] == "Pooling" || layer_types_[type_id] == "Eltwise"
			  || layer_types_[type_id] == "Split" || layer_types_[type_id] == "Scale")
		  {
			  if (diff_caches_.find(layer_types_[type_id]) == diff_caches_.end())
			  {
				  diff_caches_[layer_types_[type_id]] = vector<Blob<Dtype>*>();
				  diff_caches_used_[layer_types_[type_id]] = vector<int>();
			  }
		  }
	  }
	  used_cache_record_.resize(blobs_.size(), -1);
	  layer_type_record_.resize(blobs_.size(), "");
	  // init cache shape
	  for (int i = layers_.size() - 1; i >= 0; i--)
	  {
		  if (layer_need_backward_[i])
		  {
			  if (diff_caches_.find(layer_types_[i]) != diff_caches_.end() && bottom_need_backward_[i][0]
				  && bottom_vecs_[i][0] != top_vecs_[i][0])
			  {
				  int find_cache_id = -1;
				  //vector<int> cache_used;
				  // get cache according to layer type.
				  //vector<Blob<Dtype>*> diff_cache = diff_caches_[layer_types_[i]];
				  //cache_used = diff_caches_used_[layer_types_[i]];

				  int share_bottom_id = -1;
				  if (layer_types_[i] == "Eltwise")
				  {
					  share_bottom_id = 1;
				  }
				  else
				  {
					  share_bottom_id = 0;
				  }
				  //search free cache
				  for (int ch_id = 0; ch_id < diff_caches_[layer_types_[i]].size(); ch_id++)
				  {
					  if (diff_caches_used_[layer_types_[i]][ch_id] < 0)
					  {
						  find_cache_id = ch_id;
					  }

				  }
				  if (find_cache_id >= 0 && find_cache_id < diff_caches_[layer_types_[i]].size())
				  {
					  //get cache and resize it if it's smaller
					  Blob<Dtype>* cache = diff_caches_[layer_types_[i]][find_cache_id];
					  if (cache->count() < bottom_vecs_[i][share_bottom_id]->count())
					  {
						  cache->ReshapeLike(*bottom_vecs_[i][share_bottom_id]);
						  LOG_IF(INFO, Caffe::root_solver()) << "reshape " << layer_types_[i] << " " << cache->count();
					  }
					  //bottom_vecs_[i][share_bottom_id]->ShareDiff_LE(*cache);

					  //cache_used[find_cache_id] = bottom_id_vecs_[i][share_bottom_id];
					  diff_caches_used_[layer_types_[i]][find_cache_id] = bottom_id_vecs_[i][share_bottom_id];
					  used_cache_record_[bottom_id_vecs_[i][share_bottom_id]] = find_cache_id;
					  layer_type_record_[bottom_id_vecs_[i][share_bottom_id]] = layer_types_[i];
					  //LOG(INFO) << "cache resued " << layer_types_[i] << " id " << find_cache_id;
				  }
				  else
				  {
					  //create cache.
					  Blob<Dtype>* cache = new Blob<Dtype>(
						  bottom_vecs_[i][share_bottom_id]->num(),
						  bottom_vecs_[i][share_bottom_id]->channels(),
						  bottom_vecs_[i][share_bottom_id]->height(),
						  bottom_vecs_[i][share_bottom_id]->width());

					  diff_caches_[layer_types_[i]].push_back(cache);
					  diff_caches_used_[layer_types_[i]].push_back(bottom_id_vecs_[i][share_bottom_id]);
					  //LOG(INFO) << "create cache " << layer_types_[i] << " size " << diff_caches_[layer_types_[i]].size();
					  //bottom_vecs_[i][share_bottom_id]->ShareDiff_LE(*cache);

					  find_cache_id = diff_caches_used_[layer_types_[i]].size() - 1;
					  used_cache_record_[bottom_id_vecs_[i][share_bottom_id]] = find_cache_id;
					  layer_type_record_[bottom_id_vecs_[i][share_bottom_id]] = layer_types_[i];
				  }
			  }

			  int share_bottom_id = -1;
			  if (layer_type_record_[top_id_vecs_[i][0]] == "Eltwise")
			  {
				  share_bottom_id = 1;
			  }
			  else
			  {
				  share_bottom_id = 0;
			  }
			  if (bottom_vecs_[i][share_bottom_id] != top_vecs_[i][0])
			  {
				  for (int top_id = 0; top_id < top_vecs_[i].size(); ++top_id)
				  {

					  string layer_type = layer_type_record_[top_id_vecs_[i][top_id]];
					  int used_id = used_cache_record_[top_id_vecs_[i][top_id]];
					  //LOG(INFO) << "release " << layer_type << " at " << used_id;
					  if (diff_caches_.find(layer_type) != diff_caches_.end() && used_id >= 0)
					  {
						  //if (diff_caches_used_.find(layer_type) != diff_caches_used_.end())
						  //LOG(INFO) << "layer used " << layer_type;
						  LOG(INFO) << "used cache_size " << diff_caches_used_[layer_type].size();
						  diff_caches_used_[layer_type][used_id] = -1;
					  }
					  layer_type_record_[top_id_vecs_[i][top_id]] = "";
					  used_cache_record_[top_id_vecs_[i][top_id]] = -1;
				  }
			  }
		  }
	  }
	  */
  }
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {
  NetState net_state(param.state());
  param_filtered->CopyFrom(param);
  param_filtered->clear_layer();
  for (int i = 0; i < param.layer_size(); ++i) {
    const LayerParameter& layer_param = param.layer(i);
    const string& layer_name = layer_param.name();
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
	blobs_deletable_.push_back(true);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(false);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());

	// blob usage counter, indicate that this top will use one memory. 
	// if blob_used_counter > 1 that it can relieve the memory if all bottoms return this memory.
	// if blob_used_counter == 1 that means this top is output, we should keep it's memory.
	blob_used_counter_.push_back( 1);
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  available_blobs->erase(blob_name);
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);

  //blob usage counter
  blob_used_counter_[(*blob_name_to_idx)[blob_name]] += 1;
  return blob_id;
}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  } else {
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
  param_layer_indices_.push_back(make_pair(layer_id, param_id));
  ParamSpec default_param_spec;
  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?
      &layer_param.param(param_id) : &default_param_spec;
  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    param_owners_.push_back(-1);
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());
  } else {
    // Named param blob with name we've seen before: share params
    const int owner_net_param_id = param_names_index_[param_name];
    param_owners_.push_back(owner_net_param_id);
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];
    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();
    const int param_size = layer_param.param_size();
    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
      // Permissive dimension checking -- only check counts are the same.
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);
    if (param_spec->has_lr_mult()) {
      if (has_params_lr_[learnable_param_id]) {
        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {
        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
	  if (((phase_ == Phase::TEST && opt_test_shared_memory_) || (phase_ == Phase::TRAIN && opt_memory_&& !layer_need_backward_[i]))
		  && layer_types_[i] != "Split")
	  {
		  //share cache meory.
		  for (int ti = 0; ti < top_vecs_[i].size(); ti++)
		  {
			  int blob_id = top_id_vecs_[i][ti];
			  int cache_idx = shared_blobs_index_[blob_id];
			  // if in-place, don't borrow the shared memory again.
			  if (bottom_vecs_[i].size()>ti && bottom_vecs_[i][ti] == top_vecs_[i][ti])
			  {
				  //blobs_[blob_id]->ShareData_LE(*shared_blobs_[cache_idx]);
				  continue;
			  }

			  if (shared_blobs_[cache_idx]->count() < blobs_[blob_id]->count())
			  {
				  shared_blobs_[cache_idx]->ReshapeLike(*blobs_[blob_id]);
				  LOG_IF(INFO, Caffe::root_solver()) << "reshape " << layer_types_[i] << " " << shared_blobs_[cache_idx]->count();
			  }
			  blobs_[blob_id]->ShareData_LE(*shared_blobs_[cache_idx]);
		  }
	  }
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
	  if (layer_need_backward_[i]) {
		  if (((phase_ == Phase::TEST && opt_test_shared_memory_) || (phase_ == Phase::TRAIN && opt_memory_&& !layer_need_backward_[i]))
			  && layer_types_[i] != "Split")
		  {
			  //share cache meory.
			  for (int bi = 0; bi < bottom_vecs_[i].size(); bi++)
			  {
				  int blob_id = bottom_id_vecs_[i][bi];
				  int cache_idx = shared_blobs_diff_index_[blob_id];
				  // if in-place, don't borrow the shared memory again.
				  if (top_vecs_[i].size()>bi && top_vecs_[i][bi] == bottom_vecs_[i][bi])
				  {
					  continue;
				  }

				  if (shared_blobs_diff_[cache_idx]->count() < blobs_[blob_id]->count())
				  {
					  shared_blobs_diff_[cache_idx]->ReshapeLike(*blobs_[blob_id]);
					  LOG_IF(INFO, Caffe::root_solver()) << "diff reshape " << layer_types_[i] << " " << shared_blobs_diff_[cache_idx]->count();
				  }
				  blobs_[blob_id]->ShareDiff_LE(*shared_blobs_diff_[cache_idx]);
			  }
		  }
		  /*
		  if (opt_memory_)
		  {
			  // find cache for bottom vector.
			  if (diff_caches_.find(layer_types_[i]) != diff_caches_.end() && bottom_need_backward_[i][0]
				  && bottom_vecs_[i][0]!=top_vecs_[i][0])
			  {
				  int find_cache_id = -1;
				  //vector<int> cache_used;
				  // get cache according to layer type.
				  //vector<Blob<Dtype>*> diff_cache = diff_caches_[layer_types_[i]];
				  //cache_used = diff_caches_used_[layer_types_[i]];

				  int share_bottom_id = -1;
				  if (layer_types_[i] == "Eltwise")
				  {
					  share_bottom_id = 1;
				  }
				  else
				  {
					  share_bottom_id = 0;
				  }
				  //search free cache
				  for (int ch_id = 0; ch_id < diff_caches_[layer_types_[i]].size(); ch_id++)
				  {
					  if (diff_caches_used_[layer_types_[i]][ch_id] < 0)
					  {
						  find_cache_id = ch_id;
					  }

				  }
				  if (find_cache_id >= 0 && find_cache_id < diff_caches_[layer_types_[i]].size())
				  {
					  //get cache and resize it if it's smaller
					  Blob<Dtype>* cache = diff_caches_[layer_types_[i]][find_cache_id];
					  if (cache->count() < bottom_vecs_[i][share_bottom_id]->count())
					  {
						  cache->ReshapeLike(*bottom_vecs_[i][share_bottom_id]);
						  LOG_IF(INFO, Caffe::root_solver()) << "reshape2 " << layer_types_[i] << " " << cache->count();
					  }
					  bottom_vecs_[i][share_bottom_id]->ShareDiff_LE(*cache);

					  //cache_used[find_cache_id] = bottom_id_vecs_[i][share_bottom_id];
					  diff_caches_used_[layer_types_[i]][find_cache_id] = bottom_id_vecs_[i][share_bottom_id];
					  used_cache_record_[bottom_id_vecs_[i][share_bottom_id]] = find_cache_id;
					  layer_type_record_[bottom_id_vecs_[i][share_bottom_id]] = layer_types_[i];
					  //LOG(INFO) << "cache resued " << layer_types_[i] << " id " << find_cache_id;
				  }
				  else
				  {
					  //create cache.
					  Blob<Dtype>* cache = new Blob<Dtype>(
						  bottom_vecs_[i][share_bottom_id]->num(),
						  bottom_vecs_[i][share_bottom_id]->channels(),
						  bottom_vecs_[i][share_bottom_id]->height(),
						  bottom_vecs_[i][share_bottom_id]->width());

					  diff_caches_[layer_types_[i]].push_back(cache);
					  diff_caches_used_[layer_types_[i]].push_back(bottom_id_vecs_[i][share_bottom_id]);
					  //LOG(INFO) << "create cache " << layer_types_[i] << " size " << diff_caches_[layer_types_[i]].size();
					  bottom_vecs_[i][share_bottom_id]->ShareDiff_LE(*cache);

					  find_cache_id = diff_caches_used_[layer_types_[i]].size() - 1;
					  used_cache_record_[bottom_id_vecs_[i][share_bottom_id]] = find_cache_id;
					  layer_type_record_[bottom_id_vecs_[i][share_bottom_id]] = layer_types_[i];
				  }
			  }
		  }
		  */
		  layers_[i]->Backward(
			  top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
		  if (debug_info_) { BackwardDebugInfo(i); }

		  /*
		  // release cache

		  if (opt_memory_)
		  {
			  int share_bottom_id = -1;
			  if (layer_type_record_[top_id_vecs_[i][0]] == "Eltwise")
			  {
				  share_bottom_id = 1;
			  }
			  else
			  {
				  share_bottom_id = 0;
			  }
			  if (bottom_vecs_[i][share_bottom_id] != top_vecs_[i][0])
			  {
				  for (int top_id = 0; top_id < top_vecs_[i].size(); ++top_id)
				  {

					  string layer_type = layer_type_record_[top_id_vecs_[i][top_id]];
					  int used_id = used_cache_record_[top_id_vecs_[i][top_id]];
					  //LOG(INFO) << "release " << layer_type << " at " << used_id;
					  if (diff_caches_.find(layer_type) != diff_caches_.end() && used_id >= 0)
					  {
						  //if (diff_caches_used_.find(layer_type) != diff_caches_used_.end())
						  //LOG(INFO) << "layer used " << layer_type;
						  //LOG(INFO) << "used cache_size " << diff_caches_used_[layer_type].size();
						  diff_caches_used_[layer_type][used_id] = -1;
					  }
					  layer_type_record_[top_id_vecs_[i][top_id]] = "";
					  used_cache_record_[top_id_vecs_[i][top_id]] = -1;
				  }
			  }
		  }
		  */
	  }
  }


}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
		if (target_blobs[j]->count() == source_blob.count() ||
			layers_[target_layer_id]->layer_param().force_copy())
		{
			target_blobs[j]->CopyFrom(source_blob, false, false,true);
		}
		else
		{
			LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
				<< source_layer_name << "'; shape mismatch.  Source param shape is "
				<< source_blob.shape_string() << "; target param shape is "
				<< target_blobs[j]->shape_string() << ". "
				<< "To learn this layer's parameters from scratch rather than "
				<< "copying from a saved net, rename the layer.";
		}
      }
	  else
	  {
		  const bool kReshape = false;
		  target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
	  }
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (trained_filename.size() >= 3 &&
      trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
