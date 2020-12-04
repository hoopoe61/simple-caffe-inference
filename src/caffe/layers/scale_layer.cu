#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

template <typename Dtype>
__global__ void ScaleForward(const int n, const Dtype* in,
    const Dtype* scale, const int scale_dim, const int inner_dim,
    Dtype* out) 
    {
        CUDA_KERNEL_LOOP(index, n) 
        {
            const int scale_index = (index / inner_dim) % scale_dim;
            out[index] = in[index] * scale[scale_index];
        }
    }

template <typename Dtype>
__global__ void ScaleBiasForward(const int n, const Dtype* in,
        const Dtype* scale, const Dtype* bias, const int scale_dim,
        const int inner_dim, Dtype* out)
    {
        CUDA_KERNEL_LOOP(index, n)
        {
            const int scale_index = (index / inner_dim) % scale_dim;
            out[index] = in[index] * scale[scale_index] + bias[scale_index];
        }
    }

namespace caffe
{
    template <typename Dtype>
    void ScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
            const vector<Blob<Dtype>*>& top)
    {
        const int count = top[0]->count();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* scale_data = ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        if (bias_layer_)
        {
            const Dtype* bias_data = this->blobs_[bias_param_id_]->gpu_data();
            ScaleBiasForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,
                    scale_data, bias_data, scale_dim_, inner_dim_, top_data);
        }
        else
        {
            ScaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,
                scale_data, scale_dim_, inner_dim_, top_data);
        }
    }

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);

}