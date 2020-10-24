#include "caffe/net.hpp"

namespace caffe
{
  template <typename Dtype>
  Net<Dtype>::Net(const NetParameter &param)
  {
    Init(param);
  }

  template <typename Dtype>
  Net<Dtype>::Net(const string &param_file, Phase phase, const int level, const vector<string> *stages)
  {
    //1.读取并解析prototxt
    NetParameter param;
    ReadNetParamsFromTextFileOrDie(param_file, &param);
    //2.为模型设置模式: train or test
    // Set phase, stages and level
    param.mutable_state()->set_phase(phase);
    //设置stage
    if (stages != NULL)
    {
      for (int i = 0; i < stages->size(); i++)
      {
        param.mutable_state()->add_stage((*stages)[i]);
      }
    }
    //设置level
    param.mutable_state()->set_level(level);
    Init(param);
  }

  template <typename Dtype>
  void Net<Dtype>::Init(const NetParameter &in_param)
  {
    //设置当前状态train or test
    // Set phase from the state.
    phase_ = in_param.state().phase();
    // Filter layers based on their include/exclude rules and
    // the current NetState.
    NetParameter filtered_param;
    FilterNet(in_param, &filtered_param); //根据规则NetStateRule,来动态构建网络
    cout << "Initializing net from parameters: " << endl;
    cout << filtered_param.DebugString() << endl;

    // Create a copy of filtered_param with splits added where necessary.
    NetParameter param;
    InsertSplits(filtered_param, &param); //判断哪些层需要进行分裂
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
    for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id)
    {
      // Inherit phase from net if unset.
      if (!param.layer(layer_id).has_phase())
      {
        param.mutable_layer(layer_id)->set_phase(phase_); //若该层没有定义自己的phase,则使用net的phase来进行设置
      }
      // Setup layer.
      const LayerParameter &layer_param = param.layer(layer_id);
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param)); //创建该层
      layer_names_.push_back(layer_param.name());
      cout << "Creating Layer " << layer_param.name() << endl;

      // Figure out this layer's input and output
      // 画出该层的数据流向
      for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id)
      {
        AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
      }
      int num_top = layer_param.top_size();
      for (int top_id = 0; top_id < num_top; ++top_id)
      {
        AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
        // Collect Input layer tops as Net inputs.
        if (layer_param.type() == "Input")
        {
          const int blob_id = blobs_.size() - 1;
          net_input_blob_indices_.push_back(blob_id);
          net_input_blobs_.push_back(blobs_[blob_id].get());
        }
      }
      // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
      // specified fewer than the required number (as specified by
      // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
      Layer<Dtype> *layer = layers_[layer_id].get();
      if (layer->AutoTopBlobs())
      {
        const int needed_num_top = std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
        for (; num_top < needed_num_top; ++num_top)
        {
          // Add "anonymous" top blobs -- do not modify available_blobs or
          // blob_name_to_idx as we don't want these blobs to be usable as input
          // to other layers.
          AppendTop(param, layer_id, num_top, NULL, NULL);
        }
      }
      // After this layer is connected, set it up.
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]); //调用layer.hpp中的SetUp()函数,这里传入了blob的指针
      cout << "Setting up " << layer_names_[layer_id] << endl;

      const int param_size = layer_param.param_size();
      const int num_param_blobs = layers_[layer_id]->blobs().size();
      if (!(param_size < num_param_blobs))
      {
        cout << "Too many params specified for layer " << layer_param.name() << endl;
      }
    }
    // In the end, all remaining blobs are considered output blobs.
    for (set<string>::iterator it = available_blobs.begin(); it != available_blobs.end(); ++it)
    {
      cout << "This network produces output " << *it << endl;
      net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
      net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
    }
    for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id)
    {
      blob_names_index_[blob_names_[blob_id]] = blob_id;
    }
    for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id)
    {
      layer_names_index_[layer_names_[layer_id]] = layer_id;
    }
    ShareWeights();
    cout << "Network initialization done." << endl;
  }

  template <typename Dtype>
  void Net<Dtype>::FilterNet(const NetParameter &param, NetParameter *param_filtered)
  {
    NetState net_state(param.state()); //phase,level,stage
    param_filtered->CopyFrom(param);
    param_filtered->clear_layer();
    for (int i = 0; i < param.layer_size(); ++i)
    {
      const LayerParameter &layer_param = param.layer(i);
      const string &layer_name = layer_param.name();
      if (!(layer_param.include_size() == 0 || layer_param.exclude_size() == 0))
      {
        cout << "Specify either include rules or exclude rules; not both." << endl;
      }
      // If no include rules are specified, the layer is included by default and
      // only excluded if it meets one of the exclude rules.
      bool layer_included = (layer_param.include_size() == 0); //若在prototxt文件的层中,没有包含Include参数,则默认为true
      for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j)
      {
        if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name))
        {
          layer_included = false;
        }
      }
      for (int j = 0; !layer_included && j < layer_param.include_size(); ++j)
      {
        if (StateMeetsRule(net_state, layer_param.include(j), layer_name))
        {
          layer_included = true;
        }
      }
      if (layer_included)
      {
        param_filtered->add_layer()->CopyFrom(layer_param);
      }
    }
  }

  template <typename Dtype> //判断是否特定的规则,如果不符合,则返回false
  bool Net<Dtype>::StateMeetsRule(const NetState &state, const NetStateRule &rule, const string &layer_name)
  {
    // Check whether the rule is broken due to phase.
    if (rule.has_phase())
    {
      if (rule.phase() != state.phase())
      {
        //如果当前层的phase与网络的phase不一致
        cout << "The NetState phase (" << state.phase()
             << ") differed from the phase (" << rule.phase()
             << ") specified by a rule in layer " << layer_name << endl;
        return false;
      }
    }
    // Check whether the rule is broken due to min level.
    if (rule.has_min_level())
    {
      if (state.level() < rule.min_level())
      {
        cout << "The NetState level (" << state.level()
             << ") is above the min_level (" << rule.min_level()
             << ") specified by a rule in layer " << layer_name << endl;
        return false;
      }
    }
    // Check whether the rule is broken due to max level.
    if (rule.has_max_level())
    {
      if (state.level() > rule.max_level())
      {
        cout << "The NetState level (" << state.level()
             << ") is above the max_level (" << rule.max_level()
             << ") specified by a rule in layer " << layer_name << endl;
        return false;
      }
    }
    // Check whether the rule is broken due to stage. The NetState must
    // contain ALL of the rule's stages to meet it.
    for (int i = 0; i < rule.stage_size(); ++i)
    {
      // Check that the NetState contains the rule's ith stage.
      bool has_stage = false;
      for (int j = 0; !has_stage && j < state.stage_size(); ++j)
      {
        if (rule.stage(i) == state.stage(j))
        {
          has_stage = true;
        }
      }
      if (!has_stage)
      {
        cout << "The NetState did not contain stage '" << rule.stage(i)
             << "' specified by a rule in layer " << layer_name << endl;
        return false;
      }
    }
    // Check whether the rule is broken due to not_stage. The NetState must
    // contain NONE of the rule's not_stages to meet it.
    for (int i = 0; i < rule.not_stage_size(); ++i)
    {
      // Check that the NetState contains the rule's ith not_stage.
      bool has_stage = false;
      for (int j = 0; !has_stage && j < state.stage_size(); ++j)
      {
        if (rule.not_stage(i) == state.stage(j))
        {
          has_stage = true;
        }
      }
      if (has_stage)
      {
        cout << "The NetState contained a not_stage '" << rule.not_stage(i)
             << "' specified by a rule in layer " << layer_name << endl;
        return false;
      }
    }
    return true;
  }

  // Helper for Net::Init: add a new top blob to the net.
  template <typename Dtype>
  void Net<Dtype>::AppendTop(const NetParameter &param, const int layer_id,
                             const int top_id, set<string> *available_blobs,
                             map<string, int> *blob_name_to_idx)
  {
    shared_ptr<LayerParameter> layer_param(new LayerParameter(param.layer(layer_id)));
    const string &blob_name = (layer_param->top_size() > top_id) ? layer_param->top(top_id) : "(automatic)";
    // Check if we are doing in-place computation
    if (blob_name_to_idx && layer_param->bottom_size() > top_id && blob_name == layer_param->bottom(top_id))
    {
      // In-place computation
      cout << layer_param->name() << " -> " << blob_name << " (in-place)" << endl;
      ;
      top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
      top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    }
    else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end())
    {
      // If we are not doing in-place computation but have duplicated blobs,
      // raise an error.
      cout << "Top blob '" << blob_name << "' produced by multiple sources." << endl;
      throw;
    }
    else
    {
      // Normal output.
      cout << layer_param->name() << " -> " << blob_name << endl;
      shared_ptr<Blob<Dtype>> blob_pointer(new Blob<Dtype>());
      const int blob_id = blobs_.size();
      blobs_.push_back(blob_pointer);
      blob_names_.push_back(blob_name);
      blob_need_backward_.push_back(false);
      if (blob_name_to_idx)
      {
        (*blob_name_to_idx)[blob_name] = blob_id;
      }
      top_id_vecs_[layer_id].push_back(blob_id);
      top_vecs_[layer_id].push_back(blob_pointer.get());
    }
    if (available_blobs)
    {
      available_blobs->insert(blob_name);
    }
  }

  // Helper for Net::Init: add a new bottom blob to the net.
  template <typename Dtype>
  int Net<Dtype>::AppendBottom(const NetParameter &param, const int layer_id,
                               const int bottom_id, set<string> *available_blobs,
                               map<string, int> *blob_name_to_idx)
  {
    const LayerParameter &layer_param = param.layer(layer_id);
    const string &blob_name = layer_param.bottom(bottom_id); //bottom的名字,这是上一层的内容,在AppendTop中已经创建过
    if (available_blobs->find(blob_name) == available_blobs->end())
    {
      cout << "Unknown bottom blob '" << blob_name << "' (layer '"
           << layer_param.name() << "', bottom index " << bottom_id << ")" << endl;
      throw;
    }
    const int blob_id = (*blob_name_to_idx)[blob_name];
    cout << layer_names_[layer_id] << " <- " << blob_name << endl;
    bottom_vecs_[layer_id].push_back(blobs_[blob_id].get()); //压入已经在AppendTop中初始化过的指针
    bottom_id_vecs_[layer_id].push_back(blob_id);
    available_blobs->erase(blob_name);
    bottom_need_backward_[layer_id].push_back(false);
    return blob_id;
  }

  //TODO 这里可以把传入的参数给去掉
  template <typename Dtype>
  const vector<Blob<Dtype> *> &Net<Dtype>::Forward(Dtype *loss)
  {
    int start = 0, end = layers_.size() - 1;
    for (int i = start; i <= end; ++i)
    {
      for (int c = 0; c < before_forward_.size(); ++c)
      {
        before_forward_[c]->run(i);
      }
      layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
      for (int c = 0; c < after_forward_.size(); ++c)
      {
        after_forward_[c]->run(i);
      }
    }
    return net_output_blobs_;
  }

  template <typename Dtype>
  const vector<Blob<Dtype> *> &Net<Dtype>::Forward(
      const vector<Blob<Dtype> *> &bottom, Dtype *loss)
  {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
                               << "will be removed in a future version. Use Forward(loss).";
    // Copy bottom to net bottoms
    for (int i = 0; i < bottom.size(); ++i)
    {
      net_input_blobs_[i]->CopyFrom(*bottom[i]);
    }
    return Forward(loss);
  }

  template <typename Dtype>
  void Net<Dtype>::ForwardDebugInfo(const int layer_id)
  {
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id)
    {
      const Blob<Dtype> &blob = *top_vecs_[layer_id][top_id];
      const string &blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
          << "    [Forward] "
          << "Layer " << layer_names_[layer_id]
          << ", top blob " << blob_name
          << " data: " << data_abs_val_mean;
    }
    for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
         ++param_id)
    {
      const Blob<Dtype> &blob = *layers_[layer_id]->blobs()[param_id];
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      const string &blob_name = param_display_names_[net_param_id];
      const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
      LOG_IF(INFO, Caffe::root_solver())
          << "    [Forward] "
          << "Layer " << layer_names_[layer_id]
          << ", param blob " << blob_name
          << " data: " << data_abs_val_mean;
    }
  }

  template <typename Dtype>
  void Net<Dtype>::ShareTrainedLayersWith(const Net *other)
  {
    int num_source_layers = other->layers().size();
    for (int i = 0; i < num_source_layers; ++i)
    {
      Layer<Dtype> *source_layer = other->layers()[i].get();
      const string &source_layer_name = other->layer_names()[i];
      int target_layer_id = 0;
      while (target_layer_id != layer_names_.size() &&
             layer_names_[target_layer_id] != source_layer_name)
      {
        ++target_layer_id;
      }
      if (target_layer_id == layer_names_.size())
      {
        LOG(INFO) << "Ignoring source layer " << source_layer_name;
        continue;
      }
      DLOG(INFO) << "Copying source layer " << source_layer_name;
      vector<shared_ptr<Blob<Dtype>>> &target_blobs =
          layers_[target_layer_id]->blobs();
      CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
          << "Incompatible number of blobs for layer " << source_layer_name;
      for (int j = 0; j < target_blobs.size(); ++j)
      {
        Blob<Dtype> *source_blob = source_layer->blobs()[j].get();
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
  void Net<Dtype>::Reshape()
  {
    for (int i = 0; i < layers_.size(); ++i)
    {
      layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
    }
  }

  template <typename Dtype>
  void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter &param)
  {
    int num_source_layers = param.layer_size();
    for (int i = 0; i < num_source_layers; ++i)
    {
      const LayerParameter &source_layer = param.layer(i);
      const string &source_layer_name = source_layer.name();
      int target_layer_id = 0;
      while (target_layer_id != layer_names_.size() &&
             layer_names_[target_layer_id] != source_layer_name)
      {
        ++target_layer_id;
      }
      if (target_layer_id == layer_names_.size())
      {
        LOG(INFO) << "Ignoring source layer " << source_layer_name;
        continue;
      }
      DLOG(INFO) << "Copying source layer " << source_layer_name;
      vector<shared_ptr<Blob<Dtype>>> &target_blobs =
          layers_[target_layer_id]->blobs();
      CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
          << "Incompatible number of blobs for layer " << source_layer_name;
      for (int j = 0; j < target_blobs.size(); ++j)
      {
        if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j)))
        {
          Blob<Dtype> source_blob;
          const bool kReshape = true;
          source_blob.FromProto(source_layer.blobs(j), kReshape);
          LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
                     << source_layer_name << "'; shape mismatch.  Source param shape is "
                     << source_blob.shape_string() << "; target param shape is "
                     << target_blobs[j]->shape_string() << ". "
                     << "To learn this layer's parameters from scratch rather than "
                     << "copying from a saved net, rename the layer.";
        }
        const bool kReshape = false;
        target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
      }
    }
  }

  template <typename Dtype>
  void Net<Dtype>::CopyTrainedLayersFrom(const string &trained_filename)
  {
    if (H5Fis_hdf5(trained_filename.c_str()))
    {
      CopyTrainedLayersFromHDF5(trained_filename);
    }
    else
    {
      CopyTrainedLayersFromBinaryProto(trained_filename);
    }
  }

  template <typename Dtype>
  void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
      const string &trained_filename)
  {
    NetParameter param;
    ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
    CopyTrainedLayersFrom(param);
  }

  template <typename Dtype>
  void Net<Dtype>::CopyTrainedLayersFromHDF5(const string &trained_filename)
  {
#ifdef USE_HDF5
    hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                             H5P_DEFAULT);
    CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
    hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
    CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
    int num_layers = hdf5_get_num_links(data_hid);
    for (int i = 0; i < num_layers; ++i)
    {
      string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
      if (!layer_names_index_.count(source_layer_name))
      {
        LOG(INFO) << "Ignoring source layer " << source_layer_name;
        continue;
      }
      int target_layer_id = layer_names_index_[source_layer_name];
      DLOG(INFO) << "Copying source layer " << source_layer_name;
      vector<shared_ptr<Blob<Dtype>>> &target_blobs =
          layers_[target_layer_id]->blobs();
      hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
                                 H5P_DEFAULT);
      CHECK_GE(layer_hid, 0)
          << "Error reading weights from " << trained_filename;
      // Check that source layer doesn't have more params than target layer
      int num_source_params = hdf5_get_num_links(layer_hid);
      CHECK_LE(num_source_params, target_blobs.size())
          << "Incompatible number of blobs for layer " << source_layer_name;
      for (int j = 0; j < target_blobs.size(); ++j)
      {
        ostringstream oss;
        oss << j;
        string dataset_name = oss.str();
        int target_net_param_id = param_id_vecs_[target_layer_id][j];
        if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT))
        {
          // Target param doesn't exist in source weights...
          if (param_owners_[target_net_param_id] != -1)
          {
            // ...but it's weight-shared in target, so that's fine.
            continue;
          }
          else
          {
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
#else
    LOG(FATAL) << "CopyTrainedLayersFromHDF5 requires hdf5;"
               << " compile with USE_HDF5.";
#endif // USE_HDF5
  }

  template <typename Dtype>
  void Net<Dtype>::ToProto(NetParameter *param, bool write_diff) const
  {
    param->Clear();
    param->set_name(name_);
    // Add bottom and top
    DLOG(INFO) << "Serializing " << layers_.size() << " layers";
    for (int i = 0; i < layers_.size(); ++i)
    {
      LayerParameter *layer_param = param->add_layer();
      layers_[i]->ToProto(layer_param, write_diff);
    }
  }

  template <typename Dtype>
  void Net<Dtype>::ToHDF5(const string &filename, bool write_diff) const
  {
// This code is taken from https://github.com/sh1r0/caffe-android-lib
#ifdef USE_HDF5
    hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                               H5P_DEFAULT);
    CHECK_GE(file_hid, 0)
        << "Couldn't open " << filename << " to save weights.";
    hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
                                H5P_DEFAULT);
    CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
    hid_t diff_hid = -1;
    if (write_diff)
    {
      diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
                            H5P_DEFAULT);
      CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
    }
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id)
    {
      const LayerParameter &layer_param = layers_[layer_id]->layer_param();
      string layer_name = layer_param.name();
      hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
                                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_data_hid, 0)
          << "Error saving weights to " << filename << ".";
      hid_t layer_diff_hid = -1;
      if (write_diff)
      {
        layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
                                    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_GE(layer_diff_hid, 0)
            << "Error saving weights to " << filename << ".";
      }
      int num_params = layers_[layer_id]->blobs().size();
      for (int param_id = 0; param_id < num_params; ++param_id)
      {
        ostringstream dataset_name;
        dataset_name << param_id;
        const int net_param_id = param_id_vecs_[layer_id][param_id];
        if (param_owners_[net_param_id] == -1)
        {
          // Only save params that own themselves
          hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
                                      *params_[net_param_id]);
        }
        if (write_diff)
        {
          // Write diffs regardless of weight-sharing
          hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
                                      *params_[net_param_id], true);
        }
      }
      H5Gclose(layer_data_hid);
      if (write_diff)
      {
        H5Gclose(layer_diff_hid);
      }
    }
    H5Gclose(data_hid);
    if (write_diff)
    {
      H5Gclose(diff_hid);
    }
    H5Fclose(file_hid);
// This code is taken from https://github.com/sh1r0/caffe-android-lib
#else
    LOG(FATAL) << "ToHDF5 requires hdf5; compile with USE_HDF5.";
#endif // USE_HDF5
  }

  template <typename Dtype>
  void Net<Dtype>::ShareWeights()
  {
    for (int i = 0; i < params_.size(); ++i)
    {
      if (param_owners_[i] < 0)
      {
        continue;
      }
      params_[i]->ShareData(*params_[param_owners_[i]]);
      params_[i]->ShareDiff(*params_[param_owners_[i]]);
    }
  }

  template <typename Dtype>
  bool Net<Dtype>::has_blob(const string &blob_name) const
  {
    return blob_names_index_.find(blob_name) != blob_names_index_.end();
  }

  template <typename Dtype>
  const shared_ptr<Blob<Dtype>> Net<Dtype>::blob_by_name(
      const string &blob_name) const
  {
    shared_ptr<Blob<Dtype>> blob_ptr;
    if (has_blob(blob_name))
    {
      blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
    }
    else
    {
      blob_ptr.reset((Blob<Dtype> *)(NULL));
      LOG(WARNING) << "Unknown blob name " << blob_name;
    }
    return blob_ptr;
  }

  template <typename Dtype>
  bool Net<Dtype>::has_layer(const string &layer_name) const
  {
    return layer_names_index_.find(layer_name) != layer_names_index_.end();
  }

  template <typename Dtype>
  const shared_ptr<Layer<Dtype>> Net<Dtype>::layer_by_name(
      const string &layer_name) const
  {
    shared_ptr<Layer<Dtype>> layer_ptr;
    if (has_layer(layer_name))
    {
      layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
    }
    else
    {
      layer_ptr.reset((Layer<Dtype> *)(NULL));
      LOG(WARNING) << "Unknown layer name " << layer_name;
    }
    return layer_ptr;
  }

  INSTANTIATE_CLASS(Net);

} // namespace caffe
