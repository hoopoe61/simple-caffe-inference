#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top)
    {
        const Dtype *bottom_data = bottom[0]->gpu_data();
        Dtype *top_data = top[0]->mutable_gpu_data();
        int num = bottom[0]->shape(0);
        int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0) * channels_); //相当于bottom[0]->count(2)

        if (bottom[0] != top[0])
        {
            caffe_copy(bottom[0]->count(), bottom_data, top_data); //内置选择cpu模式还是gpu模式
        }

        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ? 0 : 1 / this->blobs_[2]->cpu_data()[0];//TODO(dengshunge) 能否换成gpu_data()?
        caffe_gpu_scale(mean_.count(), scale_factor,
                        this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
        caffe_gpu_scale(variance_.count(), scale_factor,
                        this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());

        // subtract mean
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                              batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
                              num_by_chans_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                              spatial_dim, 1, -1, num_by_chans_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), 1., top_data);

        // normalize variance
        caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
        caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(),
                   variance_.mutable_gpu_data());

        // replicate variance to input size
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                              batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
                              num_by_chans_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                              spatial_dim, 1, 1., num_by_chans_.gpu_data(),
                              spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
        caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
    }

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);
}