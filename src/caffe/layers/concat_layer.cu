#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    __global__ void Concat(const int nthreads, const Dtype* in_data,
            const bool forward, const int num_concats, const int concat_size,
            const int top_concat_axis, const int bottom_concat_axis,
            const int offset_concat_axis, Dtype* out_data)
    {
        CUDA_KERNEL_LOOP(index, nthreads)
        {
            const int total_concat_size = concat_size * bottom_concat_axis;
            const int concat_num = index / total_concat_size;
            const int concat_index = index % total_concat_size;
            const int top_index = concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
            out_data[top_index] = in_data[index];
        }
    }

    template <typename Dtype>
    void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top)
    {
        if (bottom.size() == 1)
        {
            return;
        }
        Dtype* top_data = top[0]->mutable_gpu_data();
        int offset_concat_axis = 0;
        const int top_concat_axis = top[0]->shape(concat_axis_);
        for (int i = 0; i < bottom.size(); ++i)
        {
            const Dtype* bottom_data = bottom[i]->gpu_data();
            const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
            const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
            const int nthreads = bottom_concat_size * num_concats_;

            Concat<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
                nthreads, bottom_data, true, num_concats_, concat_input_size_,
                top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data);

            offset_concat_axis += bottom_concat_axis;
        }

    }

INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);
}