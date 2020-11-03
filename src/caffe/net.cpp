#include "caffe/net.hpp"

namespace caffe
{
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
        top_id_vecs_.resize(param.layer_size());
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
            // input层的bottom.size()=0,所以会从appendTop开始
            // 因此,下面appendTop()和AppendBottom()的逻辑为:
            // 1.通过AppendTop()来创建blob指针,及保存起blob_name和blob_id
            // 2.然后在下一层的AppendBottom()中根据blob_name来查找到top中对应的blob,以此来关联起来
            for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id)
            {
                AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
            }
            int num_top = layer_param.top_size();
            for (int top_id = 0; top_id < num_top; ++top_id)
            {
                AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
                // 收集输入层作为输入,此时输入层的类型必须为"Input"
                if (layer_param.type() == "Input")
                {
                    const int blob_id = blobs_.size() - 1;
                    net_input_blob_indices_.push_back(blob_id);
                    net_input_blobs_.push_back(blobs_[blob_id].get());
                }
            }
            // 满足对top blob的需求
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
                    AppendTop(param, layer_id, num_top, nullptr, nullptr);
                }
            }
            // 当该层连接完成后,创建该层
            layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]); //调用layer.hpp中的SetUp()函数,这里传入了blob的指针
            // LOG_IF(INFO, Caffe::root_solver())
            //     << "Setting up " << layer_names_[layer_id];
        }
        // 将剩下的blobs认为是输出层
        // 在AppendBottom()中erase相应的blob_name,在AppendTop()中插入相应的blob_name
        // 因此,在输出层中,最后执行的是AppendTop(),会剩下blob_name,即对应输出层
        for (set<string>::iterator it = available_blobs.begin(); it != available_blobs.end(); ++it)
        {
            LOG_IF(INFO, Caffe::root_solver())
                << "This network produces output " << *it;
            net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
            net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
        }
        LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
    }

    template <typename Dtype>
    void Net<Dtype>::FilterNet(const NetParameter &param, NetParameter *param_filtered)
    {
        NetState net_state(param.state()); //phase,level,stage
        param_filtered->CopyFrom(param);
        param_filtered->clear_layer(); //清楚所有层的信息,但会保留模型的相关信息
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
                //将模型增加该层的参数
                param_filtered->add_layer()->CopyFrom(layer_param);
            }
        }
    }

    template <typename Dtype>
    bool Net<Dtype>::StateMeetsRule(const NetState &state,
                                    const NetStateRule &rule, const string &layer_name)
    {
        // Check whether the rule is broken due to phase.
        if (rule.has_phase())
        {
            //当前层的phase与模型的phase不一致
            if (rule.phase() != state.phase())
            {
                LOG_IF(INFO, Caffe::root_solver())
                    << "The NetState phase (" << state.phase()
                    << ") differed from the phase (" << rule.phase()
                    << ") specified by a rule in layer " << layer_name;
                return false;
            }
        }
        // Check whether the rule is broken due to min level.
        if (rule.has_min_level())
        {
            if (state.level() < rule.min_level())
            {
                LOG_IF(INFO, Caffe::root_solver())
                    << "The NetState level (" << state.level()
                    << ") is above the min_level (" << rule.min_level()
                    << ") specified by a rule in layer " << layer_name;
                return false;
            }
        }
        // Check whether the rule is broken due to max level.
        if (rule.has_max_level())
        {
            if (state.level() > rule.max_level())
            {
                LOG_IF(INFO, Caffe::root_solver())
                    << "The NetState level (" << state.level()
                    << ") is above the max_level (" << rule.max_level()
                    << ") specified by a rule in layer " << layer_name;
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
                LOG_IF(INFO, Caffe::root_solver())
                    << "The NetState did not contain stage '" << rule.stage(i)
                    << "' specified by a rule in layer " << layer_name;
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
    void Net<Dtype>::AppendTop(const NetParameter &param, const int layer_id,
                               const int top_id, set<string> *available_blobs,
                               map<string, int> *blob_name_to_idx)
    {
        shared_ptr<LayerParameter> layer_param(new LayerParameter(param.layer(layer_id)));
        const string &blob_name = (layer_param->top_size() > top_id) ? layer_param->top(top_id) : "(automatic)";
        // 检查是否是in-place操作,主要判断依据是比较top的名字是否和bottom的名字相同
        if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
            blob_name == layer_param->bottom(top_id))
        {
            // In-place computation
            // LOG_IF(INFO, Caffe::root_solver())
            //     << layer_param->name() << " -> " << blob_name << " (in-place)";
            // 在一开始,已经为top_vecs_预分配了空间
            // get()是作用于智能指针上,返回其对应的指针
            // 因为是in-place操作,之前已经保存在具有相同名字的blob,所以这里直接取对指针即可,指向同一个内存位置
            top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
            top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
        }
        else if (blob_name_to_idx &&
                 blob_name_to_idx->find(blob_name) != blob_name_to_idx->end())
        {
            // 如果不进行in-place计算,但又存在重复的blobs,则会报错
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
            blobs_.push_back(blob_pointer); //保存其智能指针
            blob_names_.push_back(blob_name);
            if (blob_name_to_idx)
            {
                (*blob_name_to_idx)[blob_name] = blob_id; //保存每个blob名字对应的id
            }
            top_vecs_[layer_id].push_back(blob_pointer.get());
            top_id_vecs_[layer_id].push_back(blob_id);
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
        const string &blob_name = layer_param.bottom(bottom_id); //bottom的名字,对应的是上一层的top,在AppendTop中已经创建过
        if (available_blobs->find(blob_name) == available_blobs->end())
        {
            // 如果之前未创建过该blob,则会报错
            LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                       << layer_param.name() << "', bottom index " << bottom_id << ")";
        }
        const int blob_id = (*blob_name_to_idx)[blob_name];
        // LOG_IF(INFO, Caffe::root_solver())
        //     << layer_names_[layer_id] << " <- " << blob_name;
        bottom_vecs_[layer_id].push_back(blobs_[blob_id].get()); //压入已经在AppendTop中初始化过的指针
        bottom_id_vecs_[layer_id].push_back(blob_id);
        available_blobs->erase(blob_name);
        return blob_id;
    }

    template <typename Dtype>
    const vector<Blob<Dtype> *> &Net<Dtype>::Forward()
    {
        for (int i = 0; i < layers_.size(); ++i)
        {
            layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
        }
        return net_output_blobs_;
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
            } //找到已经构建好的Net中对应源文件的id
            if (target_layer_id == layer_names_.size())
            {
                LOG(INFO) << "Ignoring source layer " << source_layer_name;
                continue;
            }
            DLOG(INFO) << "Copying source layer " << source_layer_name;
            vector<shared_ptr<Blob<Dtype>>> &target_blobs =
                layers_[target_layer_id]->blobs();
            CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
                << "Incompatible number of blobs for layer " << source_layer_name; //检查数量是否一致
            for (int j = 0; j < target_blobs.size(); ++j)
            { //例如conv层的size=2,分别是权重与偏置(也可能size=1)
                if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j)))
                {
                    // 如果形状不一致,输出报错信息,做如下操作,是报错更加具体
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

    INSTANTIATE_CLASS(Net);

} // namespace caffe