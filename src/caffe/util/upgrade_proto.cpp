#include "caffe/util/upgrade_proto.hpp"

namespace caffe
{
    bool UpgradeNetAsNeeded(const string &param_file, NetParameter *param)
    {
        bool success = true;
        // NetParameter uses old style input fields; try to upgrade it.
        // 更新输入层
        if (NetNeedsInputUpgrade(*param))
        {
            LOG(INFO) << "Attempting to upgrade input file specified using deprecated "
                      << "input fields: " << param_file;
            UpgradeNetInput(param);
            LOG(INFO) << "Successfully upgraded file specified using deprecated "
                      << "input fields.";
            LOG(WARNING) << "Note that future Caffe releases will only support "
                         << "input layers and not input fields.";
        }
        // NetParameter uses old style batch norm layers; try to upgrade it.
        // 更新BN层
        // TODO 只是更新学习率和衰减,对推理并无影响,应该可以删除
        if (NetNeedsBatchNormUpgrade(*param))
        {
            LOG(INFO) << "Attempting to upgrade batch norm layers using deprecated "
                      << "params: " << param_file;
            UpgradeNetBatchNorm(param);
            LOG(INFO) << "Successfully upgraded batch norm layers using deprecated "
                      << "params.";
        }
        return success;
    }

    void ReadNetParamsFromTextFileOrDie(const string &param_file,
                                        NetParameter *param)
    {
        //解析deploy文件,放在param中
        CHECK(ReadProtoFromTextFile(param_file, param))
            << "Failed to parse NetParameter file: " << param_file;
        //更新相关内容
        std::cout<<param->input_size()<<std::endl;
        UpgradeNetAsNeeded(param_file, param);
    }

    void ReadNetParamsFromBinaryFileOrDie(const string &param_file,
                                          NetParameter *param)
    {
        //解析二进制文件
        CHECK(ReadProtoFromBinaryFile(param_file, param))
            << "Failed to parse NetParameter file: " << param_file;
        UpgradeNetAsNeeded(param_file, param);
    }

    bool NetNeedsInputUpgrade(const NetParameter &net_param)
    {
        //判断是否具有一个或以上的输入层
        return net_param.input_size() > 0;
    }

    void UpgradeNetInput(NetParameter *net_param)
    {
        //更新输出层
        // Collect inputs and convert to Input layer definitions.
        // If the NetParameter holds an input alone, without shape/dim, then
        // it's a legacy caffemodel and simply stripping the input field is enough.
        bool has_shape = net_param->input_shape_size() > 0;
        bool has_dim = net_param->input_dim_size() > 0;
        if (has_shape || has_dim)
        {
            LayerParameter *layer_param = net_param->add_layer();
            layer_param->set_name("input");
            layer_param->set_type("Input");
            InputParameter *input_param = layer_param->mutable_input_param();
            // Convert input fields into a layer.
            // 得到一个input的layer？是因为最底层的input是一个独立的存在？
            for (int i = 0; i < net_param->input_size(); ++i)
            {
                layer_param->add_top(net_param->input(i));
                if (has_shape)
                {
                    input_param->add_shape()->CopyFrom(net_param->input_shape(i));
                }
                else
                {
                    // Turn legacy input dimensions into shape.
                    BlobShape *shape = input_param->add_shape();
                    int first_dim = i * 4;
                    int last_dim = first_dim + 4;
                    for (int j = first_dim; j < last_dim; j++)
                    {
                        shape->add_dim(net_param->input_dim(j));
                    }
                }
            }
            // Swap input layer to beginning of net to satisfy layer dependencies.
            for (int i = net_param->layer_size() - 1; i > 0; --i)
            {
                net_param->mutable_layer(i - 1)->Swap(net_param->mutable_layer(i));
            }
        }
        // Clear inputs.
        net_param->clear_input();
        net_param->clear_input_shape();
        net_param->clear_input_dim();
    }

    bool NetNeedsBatchNormUpgrade(const NetParameter &net_param)
    {
        for (int i = 0; i < net_param.layer_size(); ++i)
        {
            // Check if BatchNorm layers declare three parameters, as required by
            // the previous BatchNorm layer definition.
            if (net_param.layer(i).type() == "BatchNorm" && net_param.layer(i).param_size() == 3)
            {
                return true;
            }
        }
        return false;
    }

    void UpgradeNetBatchNorm(NetParameter *net_param)
    {
        for (int i = 0; i < net_param->layer_size(); ++i)
        {
            // Check if BatchNorm layers declare three parameters, as required by
            // the previous BatchNorm layer definition.
            if (net_param->layer(i).type() == "BatchNorm" && net_param->layer(i).param_size() == 3)
            {
                // set lr_mult and decay_mult to zero. leave all other param intact.
                for (int ip = 0; ip < net_param->layer(i).param_size(); ip++)
                {
                    ParamSpec *fixed_param_spec = net_param->mutable_layer(i)->mutable_param(ip);
                    fixed_param_spec->set_lr_mult(0.f);
                    fixed_param_spec->set_decay_mult(0.f);
                }
            }
        }
    }

} // namespace caffe