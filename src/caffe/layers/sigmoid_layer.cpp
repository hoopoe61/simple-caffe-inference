#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe
{

    template <typename Dtype>
    inline Dtype sigmoid(Dtype x)
    {
        return 0.5 * tanh(0.5 * x) + 0.5;
    }

    template <typename Dtype>
    void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top)
    {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        for (int i = 0; i < count; ++i)
        {
            top_data[i] = sigmoid(bottom_data[i]);
        }
    }


    INSTANTIATE_CLASS(SigmoidLayer);
    REGISTER_LAYER_CLASS(Sigmoid);

} // namespace caffe
