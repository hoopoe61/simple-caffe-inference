#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe 
{
    template <typename Dtype>
    __global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) 
    {
        CUDA_KERNEL_LOOP(index, n) 
        {
            out[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
        }
    }

    template <typename Dtype>
    void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) 
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();

        SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                count, bottom_data, top_data);

        CUDA_POST_KERNEL_CHECK;
    }

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLayer);
}  // namespace caffe
