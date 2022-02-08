#include "caffe/util/insert_splits.hpp"

namespace caffe
{
    /**
   * 注意,存在如下几组关系:
   * 1.data层没有bottom blob,只有top blob;
   * 2.第i层的top blob name与第i+1层的bottom blob name相同.
   * 3.需要理解下述代码,主要了解这几种结构既可以
   *                                                       ----------
   *                                                |--->  |  conv  |  ---> ...
   * ----------      ----------      ----------     |      ----------
   * |  data  |  ->  |  conv  |  ->  |  conv  |  ---|
   * ----------      ----------      ----------     |      ----------
   *                                                |--->  |  conv  |  ---> ...
   *                                                       ----------
   * 
  */
    // split层作用：划分一个input blob给多个output blobs使用；
    void InsertSplits(const NetParameter &param, NetParameter *param_split)
    {
        // 初始化相关网络参数,并清楚层相关信息
        param_split->CopyFrom(param);
        param_split->clear_layer();

        map<string, pair<int, int>> blob_name_to_last_top_idx;            //key:blob_name ;vlue:(第i层,第j个blob).存储blob name及其对应的第i层
        map<pair<int, int>, pair<int, int>> bottom_idx_to_source_top_idx; //将bottom与top关联起来
        map<pair<int, int>, int> top_idx_to_bottom_count;                 //top blob对应的bottom blob的数量
        map<pair<int, int>, int> top_idx_to_bottom_split_idx;             //top blob切割成n份bottom时对应的idx
        map<int, string> layer_idx_to_layer_name;                         //idx与层名字

        //读取所有层的信息,建立相邻层之间blob的连接信息
        for (int i = 0; i < param.layer_size(); ++i)
        {
            const LayerParameter &layer_param = param.layer(i);
            layer_idx_to_layer_name[i] = layer_param.name();
            for (int j = 0; j < layer_param.bottom_size(); ++j)
            {
                // 一开始的data层没有bottom
                // 所以是将第i层的bottom与第i-1层的top关联起来.
                const string &blob_name = layer_param.bottom(j);
                if (blob_name_to_last_top_idx.find(blob_name) ==
                    blob_name_to_last_top_idx.end())
                {
                    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                               << layer_param.name() << "', bottom index " << j << ")";
                }
                const pair<int, int> &bottom_idx = make_pair(i, j);
                const pair<int, int> &top_idx = blob_name_to_last_top_idx[blob_name]; //这个blob与上一个top的blob
                bottom_idx_to_source_top_idx[bottom_idx] = top_idx;                   //将bottom与top关联起来
                ++top_idx_to_bottom_count[top_idx];                                   //记录该top_idx对应需要split成多少份
            }
            for (int j = 0; j < layer_param.top_size(); ++j)
            {
                const string &blob_name = layer_param.top(j);
                blob_name_to_last_top_idx[blob_name] = make_pair(i, j); //value: (第i层,第j个blob)
            }
        }

        // 判断哪些层需要创建split
        for (int i = 0; i < param.layer_size(); ++i)
        {
            // 创建原有的层
            LayerParameter *layer_param = param_split->add_layer();
            layer_param->CopyFrom(param.layer(i));
            for (int j = 0; j < layer_param->bottom_size(); ++j)
            {
                // 连接bottom到split层中
                const pair<int, int> &top_idx = bottom_idx_to_source_top_idx[make_pair(i, j)];
                const int split_count = top_idx_to_bottom_count[top_idx];
                if (split_count > 1)
                {
                    // 如果需要split成多份,
                    // 首先获取上一层的layer_name,此时的blob_name对应上一层的top_name
                    // 并将layer_name与blob_name进行组合,获得在split层得到的name
                    const string &layer_name = layer_idx_to_layer_name[top_idx.first];
                    const string &blob_name = layer_param->bottom(j);
                    layer_param->set_bottom(j, SplitBlobName(layer_name, blob_name, top_idx.second,
                                                             top_idx_to_bottom_split_idx[top_idx]++));
                }
            }
            // 创建split层
            for (int j = 0; j < layer_param->top_size(); ++j)
            {
                const pair<int, int> &top_idx = make_pair(i, j);
                const int split_count = top_idx_to_bottom_count[top_idx]; //这个top需要split成n份
                if (split_count > 1)
                {
                    // 如果需要split成多份,
                    // 则需要获取layer_name和blob_name,以此来配置新建的split层
                    const string &layer_name = layer_idx_to_layer_name[i];
                    const string &blob_name = layer_param->top(j);
                    LayerParameter *split_layer_param = param_split->add_layer();
                    ConfigureSplitLayer(layer_name, blob_name, j, split_count,
                                        0, split_layer_param);
                }
            }
        }
    }

    void ConfigureSplitLayer(const string &layer_name, const string &blob_name,
                             const int blob_idx, const int split_count, const float loss_weight,
                             LayerParameter *split_layer_param)
    {
        split_layer_param->Clear();
        split_layer_param->add_bottom(blob_name);
        split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
        split_layer_param->set_type("Split");
        for (int k = 0; k < split_count; ++k)
        {
            //因为需要split成split_count份,所以需要创建split_count个top
            split_layer_param->add_top(
                SplitBlobName(layer_name, blob_name, blob_idx, k));
        }
    }

    string SplitLayerName(const string &layer_name, const string &blob_name,
                          const int blob_idx)
    {
        ostringstream split_layer_name;
        split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
                         << "_split";
        return split_layer_name.str();
    }

    string SplitBlobName(const string &layer_name, const string &blob_name,
                         const int blob_idx, const int split_idx)
    {
        ostringstream split_blob_name;
        split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
                        << "_split_" << split_idx;
        return split_blob_name.str();
    }

} //namespace caffe