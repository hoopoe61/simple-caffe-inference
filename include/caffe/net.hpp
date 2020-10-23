#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/upgrade_proto.hpp"

namespace caffe
{
  template <typename Dtype>
  class Net
  {
    Net(const string &param_file, Phase phase, const int level = 0, const vector<string> *stage = nullptr);
    virtual ~Net() {}

    /// @brief 初始化网络.
    void Init(const NetParameter &param);

    /// @brief 前向传播.
    const vector<Blob<Dtype> *> &Forward(Dtype *loss = NULL);

    // Helpers for Init.
    /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
    static void FilterNet(const NetParameter &param,
                          NetParameter *param_filtered);
    /// @brief return whether NetState state meets NetStateRule rule
    static bool StateMeetsRule(const NetState &state, const NetStateRule &rule, const string &layer_name);

  protected:
    // Helpers for Init.
    /// @brief Append a new top blob to the net.
    void AppendTop(const NetParameter &param, const int layer_id,
                   const int top_id, set<string> *available_blobs,
                   map<string, int> *blob_name_to_idx);
    /// @brief Append a new bottom blob to the net.
    int AppendBottom(const NetParameter &param, const int layer_id,
                     const int bottom_id, set<string> *available_blobs,
                     map<string, int> *blob_name_to_idx);
    /// @brief Append a new parameter blob to the net.
    void AppendParam(const NetParameter &param, const int layer_id,
                     const int param_id);

    /// @brief The network name
    string name_;
    /// @brief The phase: TRAIN or TEST
    Phase phase_;
    /// @brief Individual layers in the net
    vector<shared_ptr<Layer<Dtype>>> layers_; //保存每一层
    vector<string> layer_names_;              //每个layer的名字
    map<string, int> layer_names_index_;      //每个layer名字的index
    vector<string> blob_names_;               //每个blob的名字
    map<string, int> blob_names_index_;       //每个blob名字的index
    /// bottom_vecs stores the vectors containing the input for each layer.
    /// They don't actually host the blobs (blobs_ does), so we simply store
    /// pointers.
    vector<vector<Blob<Dtype> *>> bottom_vecs_;
    vector<vector<int>> bottom_id_vecs_;
    vector<vector<bool>> bottom_need_backward_;
    /// top_vecs stores the vectors containing the output for each layer
    vector<vector<Blob<Dtype> *>> top_vecs_;
    vector<vector<int>> top_id_vecs_;

    vector<vector<int>> param_id_vecs_;

    /// blob indices for the input and the output of the net
    vector<int> net_input_blob_indices_;
    vector<int> net_output_blob_indices_; //网络输出层index
    vector<Blob<Dtype> *> net_input_blobs_;
    vector<Blob<Dtype> *> net_output_blobs_; //网络输出层
    /// The bytes of memory used by this net
    size_t memory_used_;
  };
} // namespace caffe

#endif //CAFFE_NET_HPP_