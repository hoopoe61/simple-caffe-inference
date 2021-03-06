#include <vector>
#include <cfloat>
#include <algorithm>

#include "thrust/device_vector.h"

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    __global__ void kernel_channel_max(const int num, const int channels,
        const int spatial_dim, const Dtype* data, Dtype* out) 
    {
        CUDA_KERNEL_LOOP(index, num * spatial_dim) 
        {
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            Dtype maxval = -FLT_MAX;
            for (int c = 0; c < channels; ++c) 
            {
                maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
            }
            out[index] = maxval;
        }
    }

    template <typename Dtype>
    __global__ void kernel_channel_subtract(const int count,
        const int num, const int channels,
        const int spatial_dim, const Dtype* channel_max, Dtype* data) 
    {
        CUDA_KERNEL_LOOP(index, count) 
        {
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            data[index] -= channel_max[n * spatial_dim + s];
        }
    }

    template <typename Dtype>
    __global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) 
    {
        CUDA_KERNEL_LOOP(index, count) 
        {
            out[index] = exp(data[index]);
        }
    }

    template <typename Dtype>
    __global__ void kernel_channel_sum(const int num, const int channels,
        const int spatial_dim, const Dtype* data, Dtype* channel_sum) 
    {
        CUDA_KERNEL_LOOP(index, num * spatial_dim) 
        {
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            Dtype sum = 0;
            for (int c = 0; c < channels; ++c) 
            {
                sum += data[(n * channels + c) * spatial_dim + s];
            }
            channel_sum[index] = sum;
        }
    }

    template <typename Dtype>
    __global__ void kernel_channel_div(const int count,
        const int num, const int channels,
        const int spatial_dim, const Dtype* channel_sum, Dtype* data) 
    {
        CUDA_KERNEL_LOOP(index, count) 
        {
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            data[index] /= channel_sum[n * spatial_dim + s];
        }
    }

    template <typename Dtype>
    void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) 
    {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        Dtype* scale_data = scale_.mutable_gpu_data();
        int count = bottom[0]->count();
        int channels = top[0]->shape(softmax_axis_);
        caffe_copy(count, bottom_data, top_data);
        // We need to subtract the max to avoid numerical issues, compute the exp,
        // and then normalize.
        // compute max
        kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
            CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
            scale_data);
        // subtract
        kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
            scale_data, top_data);
        // exponentiate
        kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data, top_data);
        // sum after exp
        kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
            CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
            scale_data);
        // divide
        kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
            scale_data, top_data);
    }

    INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxLayer);
}