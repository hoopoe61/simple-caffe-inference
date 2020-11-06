#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe
{
    template <typename Dtype>
    class Net
    {
    public:
        Net(const string &param_file, Phase phase, const int level = 0, const vector<string> *stages = nullptr);
        Net(const Net &) = delete;
        Net &operator=(const Net &) = delete;
        virtual ~Net() {}

        /// @brief Initialize a network with a NetParameter.
        void Init(const NetParameter &param);

        ///@brief Run Forward and return the result.注意,这里与原本有区别,取消了参数的传入
        const vector<Blob<Dtype> *> &Forward();

        /**
         * @brief Reshape all layers from bottom to top.
         *
         * This is useful to propagate changes to layer sizes without running
         * a forward pass, e.g. to compute output feature size.
         */
        void Reshape();

        /**
         * @brief For an already initialized net, copies the pre-trained layers from
         *        another Net.
         *        对于一个已经初始化的网络,从其它net中复制好已经训练好的层
         */
        void CopyTrainedLayersFrom(const NetParameter &param);
        void CopyTrainedLayersFrom(const string &trained_filename);
        void CopyTrainedLayersFromBinaryProto(const string &trained_filename);

        /// @brief returns the layer names 返回层名字
        inline const vector<string> &layer_names() const { return layer_names_; }
        /// @brief returns the blob names  返回blob名字
        inline const vector<string> &blob_names() const { return blob_names_; }
        /// @brief returns the blobs  返回blob的值
        inline const vector<shared_ptr<Blob<Dtype>>> &blobs() const
        {
            return blobs_;
        }
        /// @brief returns the layers  返回layer,可以查看每层的权重
        inline const vector<shared_ptr<Layer<Dtype>>> &layers() const
        {
            return layers_;
        }

        /// @brief Input and output blob numbers
        inline int num_inputs() const { return net_input_blobs_.size(); }
        inline int num_outputs() const { return net_output_blobs_.size(); }
        /// 输入的blob
        inline const vector<Blob<Dtype> *> &input_blobs() const
        {
            return net_input_blobs_;
        }
        /// 输出的blob
        inline const vector<Blob<Dtype> *> &output_blobs() const
        {
            return net_output_blobs_;
        }

        // Helpers for Init.
        /**
         * @brief Remove layers that the user specified should be excluded given the current
         *        phase, level, and stage.
         *          根据规则,移除相应层
         */
        static void FilterNet(const NetParameter &param,
                              NetParameter *param_filtered);
        /// @brief return whether NetState state meets NetStateRule rule
        static bool StateMeetsRule(const NetState &state, const NetStateRule &rule,
                                   const string &layer_name);

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

        /// @brief The network name
        string name_;
        /// @brief The phase: TRAIN or TEST
        Phase phase_;
        /// @brief Individual layers in the net
        vector<shared_ptr<Layer<Dtype>>> layers_; //保存每一层
        vector<string> layer_names_;              //每个layer的名字
        /// @brief the blobs storing intermediate results between the layer.
        vector<shared_ptr<Blob<Dtype>>> blobs_; //每层的计算结果
        vector<string> blob_names_;             //每个blob的名字
        /// bottom_vecs stores the vectors containing the input for each layer.
        /// They don't actually host the blobs (blobs_ does), so we simply store
        /// pointers.
        vector<vector<Blob<Dtype> *>> bottom_vecs_;
        vector<vector<int>> bottom_id_vecs_;
        /// top_vecs stores the vectors containing the output for each layer
        vector<vector<Blob<Dtype> *>> top_vecs_;
        vector<vector<int>> top_id_vecs_;
        /// blob indices for the input and the output of the net
        vector<int> net_input_blob_indices_;
        vector<int> net_output_blob_indices_; //网络输出层index
        vector<Blob<Dtype> *> net_input_blobs_;
        vector<Blob<Dtype> *> net_output_blobs_; //网络输出层

    }; //class Net

} //namespace caffe

#endif //CAFFE_NET_HPP_