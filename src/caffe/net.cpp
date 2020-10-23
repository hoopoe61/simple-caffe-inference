#include "caffe/net.hpp"

namespace caffe
{
  template <typename Dtype>
  Net<Dtype>::Net(const string &param_file, Phase phase, const int level, const vector<string> *stage)
  {
    NetParameter param;
    //从文件中读取,解析参数
    ReadNetParamsFromTextFileOrDie(param_file, &param);

    // Set phase, stages and level
    //设置网络的phase,每一层的stage和网络的level
    //这phase可以是train或者test,而stage和level用的比较少,通常是为了动态构建网络,添加或者删除某些层
    param.mutable_state()->set_phase(phase);
    if (stages != nullptr)
    {
      for (int i = 0; i < stages->size(); i++)
      {
        param.mutable_state()->add_stage((*stages)[i]);
      }
    }
    param.mutable_state()->set_level(level);
    //初始化网络
    Init(param);
  }

  template <typename Dtype>
  void Net<Dtype>::Init(const NetParameter &in_param)
  {
    //设置state:trair or test
    phase_ = in_param.state().phase();

    //根据规则,来引入include或者排除exclude相关的层
    NetParameter filtered_param;
    FilterNet(in_param, &filtered_param); //根据规则NetStateRule,来动态构建网络

    NetParameter param;
    InsertSplits(filtered_param, &param); //判断哪些层需要进行分裂

    name_ = param.name();
    map<string, int> blob_name_to_idx;
    set<string> available_blobs;
    memory_used_ = 0;
    // For each layer, set up its input and output 先进行预分配空间,里面存在的是指针或者int
    bottom_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    param_id_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    bottom_need_backward_.resize(param.layer_size());

    for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id)
    {
      //若该层没有定义自己的phase,则使用net的phase来进行设置
      if (!param.layer(layer_id).has_phase())
      {
        param.mutable_layer(layer_id)->set_phase(phase_); //设置每层的phase,train或者test
      }

      const LayerParameter &layer_param = param.layer(layer_id);
      layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param)); //创建该层
      layer_names_.push_back(layer_param.name());
      cout << "Creating Layer " << layer_param.name() << endl;

      for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id)
      {
        //data层的bottom数目为0,因此data层不进入这里
        const int blob_id = AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
      }

      int num_top = layer_param.top_size();
      for (int top_id = 0; top_id < num_top; ++top_id)
      {
        AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
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
      Layer<Dtype> *layer = layers_[layer_id].get(); //针对当前layer
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

      for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id)
      {
        cout << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string() << endl;
        memory_used_ += top_vecs_[layer_id][top_id]->count();
      }
      cout < < < < "Memory required for data: " << memory_used_ * sizeof(Dtype) << endl;

      const int param_size = layer_param.param_size();               //应该是prototxt中的param
      const int num_param_blobs = layers_[layer_id]->blobs().size(); //blob的数量
      if (param_size <= num_param_blobs)
      {
        cout < < < < "Too many params specified for layer " << layer_param.name() << endl;
      }

      for (int param_id = 0; param_id < num_param_blobs; ++param_id)
      {
        AppendParam(param, layer_id, param_id);
      }
    }
    // 将剩下的blobs认为是输出层
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
} // namespace caffe