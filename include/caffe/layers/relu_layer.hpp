#ifndef CAFFE_RELU_LAYER_HPP_
#define CAFFE_RELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe
{

    /**
     * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
     *        The simple max is fast to compute, and the function does not saturate.
     */
    template <typename Dtype>
    class ReLULayer : public NeuronLayer<Dtype>
    {
    public:
        /**
         * @param param provides ReLUParameter relu_param,
         *     with ReLULayer options:
         *   - negative_slope (\b optional, default 0).
         *     the value @f$ \nu @f$ by which negative values are multiplied.
         */
        explicit ReLULayer(const LayerParameter &param)
            : NeuronLayer<Dtype>(param) {}

        virtual inline const char *type() const { return "ReLU"; }

    protected:
        /**
         * @param bottom input Blob vector (length 1)
         *   -# @f$ (N \times C \times H \times W) @f$
         *      the inputs @f$ x @f$
         * @param top output Blob vector (length 1)
         *   -# @f$ (N \times C \times H \times W) @f$
         *      the computed outputs @f$
         *        y = \max(0, x)
         *      @f$ by default.  If a non-zero negative_slope @f$ \nu @f$ is provided,
         *      the computed outputs are @f$ y = \max(0, x) + \nu \min(0, x) @f$.
         */
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);
        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);
    };

} // namespace caffe

#endif // CAFFE_RELU_LAYER_HPP_
