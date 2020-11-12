#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    template <typename Dtype>
    void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top)
    {
        BatchNormParameter param = this->layer_param_.batch_norm_param();
        use_global_stats_ = true; //因为是只有inference,所以默认是true
        if (bottom[0]->num_axes() == 1)
            channels_ = 1;
        else
            channels_ = bottom[0]->shape(1);
        eps_ = param.eps(); //默认是1e-5
        if (this->blobs_.size() > 0)
        {
            LOG(INFO) << "Skipping parameter initialization";
        }
        else
        {
            this->blobs_.resize(3);
            vector<int> sz;
            sz.push_back(channels_);
            this->blobs_[0].reset(new Blob<Dtype>(sz)); //mean
            this->blobs_[1].reset(new Blob<Dtype>(sz)); //variance
            sz[0] = 1;
            this->blobs_[2].reset(new Blob<Dtype>(sz)); //moving average factor
            for (int i = 0; i < 3; ++i)
            {
                caffe_set(this->blobs_[i]->count(), Dtype(0),
                          this->blobs_[i]->mutable_cpu_data());
            }
        }
    }

    template <typename Dtype>
    void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top)
    {
        if (bottom[0]->num_axes() >= 1)
            CHECK_EQ(bottom[0]->shape(1), channels_);
        top[0]->ReshapeLike(*bottom[0]);

        vector<int> sz;
        sz.push_back(channels_);
        mean_.Reshape(sz);
        variance_.Reshape(sz);
        temp_.ReshapeLike(*bottom[0]);
        sz[0] = bottom[0]->shape(0); //batch size
        batch_sum_multiplier_.Reshape(sz);

        int spatial_dim = bottom[0]->count() / (channels_ * bottom[0]->shape(0));
        if (spatial_sum_multiplier_.num_axes() == 0 ||
            spatial_sum_multiplier_.shape(0) != spatial_dim)
        {
            sz[0] = spatial_dim;
            spatial_sum_multiplier_.Reshape(sz);
            Dtype *multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
            caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
        }

        int numbychans = channels_ * bottom[0]->shape(0);
        if (num_by_chans_.num_axes() == 0 ||
            num_by_chans_.shape(0) != numbychans)
        {
            sz[0] = numbychans;
            num_by_chans_.Reshape(sz);
            caffe_set(batch_sum_multiplier_.count(), Dtype(1),
                      batch_sum_multiplier_.mutable_cpu_data());
        }
    }

    template <typename Dtype>
    void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top)
    {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        int num = bottom[0]->shape(0);
        int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0) * channels_); //相当于bottom[0]->count(2)

        if (bottom[0] != top[0])
        {
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }

        // use the stored mean/variance estimates.
        const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ? 0 : 1 / this->blobs_[2]->cpu_data()[0];
        caffe_cpu_scale(mean_.count(), scale_factor,
                        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
        caffe_cpu_scale(variance_.count(), scale_factor,
                        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());

        // subtract mean
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                              batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
                              num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                              spatial_dim, 1, -1, num_by_chans_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), 1., top_data);

        // normalize variance
        caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
        caffe_sqrt(variance_.count(), variance_.cpu_data(),
                   variance_.mutable_cpu_data());

        // replicate variance to input size
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
                              batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
                              num_by_chans_.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
                              spatial_dim, 1, 1., num_by_chans_.cpu_data(),
                              spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
        caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
    }

    INSTANTIATE_CLASS(BatchNormLayer);
    REGISTER_LAYER_CLASS(BatchNorm);
} // namespace caffe
