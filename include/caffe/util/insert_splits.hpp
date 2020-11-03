#ifndef CAFFE_UTIL_INSERT_SPLITS_HPP_
#define CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
    // Copy NetParameters with SplitLayers added to replace any shared bottom
    // blobs with unique bottom blobs provided by the SplitLayer.
    void InsertSplits(const NetParameter &param, NetParameter *param_split);

    // 配置split层相关参数
    void ConfigureSplitLayer(const string &layer_name, const string &blob_name,
                             const int blob_idx, const int split_count, const float loss_weight,
                             LayerParameter *split_layer_param);

    // 创建split层名字
    string SplitLayerName(const string &layer_name, const string &blob_name,
                          const int blob_idx);

    // 创建blob名字
    string SplitBlobName(const string &layer_name, const string &blob_name,
                         const int blob_idx, const int split_idx);

} // namespace caffe

#endif //CAFFE_UTIL_INSERT_SPLITS_HPP_