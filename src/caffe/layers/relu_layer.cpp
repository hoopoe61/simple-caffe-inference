#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe
{

    template <typename Dtype>
    void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top)
    {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope(); //默认是0
        for (int i = 0; i < count; ++i)
        {
            top_data[i] = std::max(bottom_data[i], Dtype(0)) + negative_slope * std::min(bottom_data[i], Dtype(0)); //可以转换为pRelu
        }
    }

    INSTANTIATE_CLASS(ReLULayer);
    REGISTER_LAYER_CLASS(ReLU);

} // namespace caffe
