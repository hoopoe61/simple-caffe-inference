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
    // 1.读取并解析prototxt
    NetParameter param;
    ReadNetParamsFromTextFileOrDie(param_file, &param);

    // 2.设置模型的phase,stage和level
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
    // 1.获取当前模型的phase(train or test)
    phase_ = in_param.state().phase();

    // 2.根据每层的include/exclude规则和当前模型的phase,来构建网络
    NetParameter filtered_param;
    FilterNet(in_param, &filtered_param);
    // LOG_IF(INFO, Caffe::root_solver())
    //     << "Initializing net from parameters: " << std::endl
    //     << filtered_param.DebugString(); //输出网络prototxt的内容

    // 3.增加splits层
    NetParameter param;
    InsertSplits(filtered_param, &param); //TODO 里面具有可优化的空间

    // 4.建立所有层,并进行连接
    name_ = param.name();
    map<string, int> blob_name_to_idx;
    set<string> available_blobs;
    // 预分配空间
    bottom_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    param_id_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    bottom_need_backward_.resize(param.layer_size());//TODO 应该可以删去
    for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id)
    {
      // 设置当前层的phase为test
      param.mutable_layer(layer_id)->set_phase(phase_);

      // 创建层
      const LayerParameter &layer_param = param.layer(layer_id);
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param)); //调用工厂模式来创建
      layer_names_.push_back(layer_param.name());
      // LOG_IF(INFO, Caffe::root_solver())
      //     << "Creating Layer " << layer_param.name();

      // 连接该层的输入输出,即将bottom与top关联起来
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

      // 当该层连接完成后,创建该层
      layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]); //调用layer.hpp中的SetUp()函数,这里传入了blob的指针
      // LOG_IF(INFO, Caffe::root_solver())
      //     << "Setting up " << layer_names_[layer_id];
    }

    // 将剩下的blobs认为是输出层
    for (set<string>::iterator it = available_blobs.begin(); it != available_blobs.end(); ++it)
    {
      LOG_IF(INFO, Caffe::root_solver())
          << "This network produces output " << *it;
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
    LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
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
      // 不能同时设置include和exclude
      CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
      // 如果该层的include规则没有被指定,则默认是include,只有遇到exclude规则是,才将其排除在外
      bool layer_included = (layer_param.include_size() == 0); //若在prototxt文件的层中,没有包含Include参数,则默认为true
      for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j)
      {
        // 当没有include,但存在exclude规则时
        if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name))
        {
          layer_included = false;
        }
      }
      for (int j = 0; !layer_included && j < layer_param.include_size(); ++j)
      {
        //
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

  // 判断是否特定的规则,如果不符合,则返回false
  template <typename Dtype>
  bool Net<Dtype>::StateMeetsRule(const NetState &state,
                                  const NetStateRule &rule, const string &layer_name)
  {
    // 如果当前层的的规则与模型的phase不一致,则返回false
    if (rule.has_phase() && (rule.phase() != state.phase()))
    {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState phase (" << state.phase()
          << ") differed from the phase (" << rule.phase()
          << ") specified by a rule in layer " << layer_name;
      return false;
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
    if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
        blob_name == layer_param->bottom(top_id))
    {
      // In-place computation
      // LOG_IF(INFO, Caffe::root_solver())
      //     << layer_param->name() << " -> " << blob_name << " (in-place)";
      top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
      top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    }
    else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end())
    {
      // If we are not doing in-place computation but have duplicated blobs,
      // raise an error.
      LOG(FATAL) << "Top blob '" << blob_name
                 << "' produced by multiple sources.";
    }
    else
    {
      // Normal output.
      // if (Caffe::root_solver())
      // {
      //   LOG(INFO) << layer_param->name() << " -> " << blob_name;
      // }
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
      LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                 << layer_param.name() << "', bottom index " << bottom_id << ")";
    }
    const int blob_id = (*blob_name_to_idx)[blob_name];
    // LOG_IF(INFO, Caffe::root_solver())
    //     << layer_names_[layer_id] << " <- " << blob_name;
    bottom_vecs_[layer_id].push_back(blobs_[blob_id].get()); //压入已经在AppendTop中初始化过的指针
    bottom_id_vecs_[layer_id].push_back(blob_id);
    available_blobs->erase(blob_name);
    bool need_backward = blob_need_backward_[blob_id];
    // Check if the backpropagation on bottom_id should be skipped
    if (layer_param.propagate_down_size() > 0)
    {
      need_backward = layer_param.propagate_down(bottom_id);
    }
    bottom_need_backward_[layer_id].push_back(need_backward);
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
    CopyTrainedLayersFromBinaryProto(trained_filename);
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
  bool Net<Dtype>::has_layer(const string &layer_name) const
  {
    return layer_names_index_.find(layer_name) != layer_names_index_.end();
  }

  INSTANTIATE_CLASS(Net);

} // namespace caffe
