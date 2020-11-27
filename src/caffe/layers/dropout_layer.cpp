// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top)
    {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        caffe_copy(bottom[0]->count(), bottom_data, top_data); //TODO 应该更加高效,直接利用共享指针
    }

    INSTANTIATE_CLASS(DropoutLayer);
    REGISTER_LAYER_CLASS(Dropout);

} // namespace caffe