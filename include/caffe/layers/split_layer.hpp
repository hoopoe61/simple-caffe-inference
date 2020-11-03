#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
    /**
     * @brief Creates a "split" path in the network by copying the bottom Blob
     *        into multiple top Blob%s to be used by multiple consuming layers.
     */
    template <typename Dtype>
    class SplitLayer : public Layer<Dtype>
    {
    public:
        explicit SplitLayer(const LayerParameter &param)
            : Layer<Dtype> (param) {}
        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const { return "Split"; }
        //bottom需要的准确数目是1
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        //最小的top blob的最小数目是1
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        int count_;

    }; //class SplitLayer

} // namespace caffe

#endif //CAFFE_SPLIT_LAYER_HPP_